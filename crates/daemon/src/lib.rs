//! The dictation engine as a library: everything `dictation-daemon`'s
//! console binary does, minus the `main()` shell -- so a GUI front end
//! (`tray-app`) can drive the exact same pipeline and just add a
//! different way of showing state and taking input.
//!
//! Wires the ring buffer, mic capture, push-to-talk + hands-free hotkeys,
//! VAD-gated streaming Whisper transcription, an optional deadlined
//! cleanup pass, and clipboard text injection together per
//! `dictation-architecture.md`. See the doc comments below for the
//! per-feature rationale (§2.1 pre-roll, §2.2 VAD/hotkeys, §2.3 streaming
//! ASR, §2.4 cleanup, §2.5 injection) -- this is the same engine described
//! in the top-level README's "Status" checklist, just split into a
//! library.
//!
//! [`Engine::load`] does all the startup work (load models, open the mic,
//! install the global hotkey hook) and returns a driveable [`Engine`].
//! [`Engine::run`] blocks, running the pipeline until a
//! [`ControlEvent::Quit`] arrives, emitting [`PipelineStatus`] updates
//! on the given channel as it goes -- that channel is how a GUI front end
//! finds out what's happening without scraping stdout. [`Engine::control_sender`]
//! is how a GUI (or anything else) injects events like a UI "quit" button
//! or a tray menu's hands-free toggle, using the exact same event type the
//! real hotkey listener uses internally.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use hotkey::{HotkeyEvent, HotkeySlot, MultiHotkeyConfig};
use ring_buffer::Mark;

/// §2.1: "capture the ~500 ms *before* the hotkey press. Solves the
/// universal 'I started talking a beat too early and lost my first word'
/// problem." The ring buffer's always-on capture is what makes this
/// possible at all -- there's no re-opening a device to catch up on.
pub const PRE_ROLL_DURATION: Duration = Duration::from_millis(500);

/// How often the worker checks for newly-ready VAD frames / decode
/// windows while a session is active. Small enough that a window becomes
/// available for decode close to the moment its last sample lands, large
/// enough not to spin the loop pointlessly.
pub const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// How often the push-to-talk watchdog cross-checks real OS key state
/// against the event-driven `ChordDetector`'s idea of whether the chord
/// is still held. See [`hotkey::is_physically_down`]'s doc comment for
/// the real, reported bug this exists to catch: a Windows global-hook
/// limitation can silently drop a `KeyRelease` event, which without this
/// leaves a session stuck open until the *next* physical press --
/// looking exactly like push-to-talk had turned into a toggle. Small
/// enough that the extra latency in the rare case this actually fires is
/// barely noticeable; large enough not to spin a thread pointlessly.
const PTT_WATCHDOG_INTERVAL: Duration = Duration::from_millis(150);

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("couldn't load ASR model: {0}")]
    AsrModel(#[from] asr::AsrError),
    #[error("couldn't load VAD model: {0}")]
    VadModel(#[from] vad::VadError),
    #[error("couldn't initialize text injection: {0}")]
    TextInjector(#[from] inject::InjectError),
    #[error("couldn't start microphone capture: {0}")]
    MicrophoneCapture(#[from] audio_input::AudioCaptureError),
}

/// Structured, cross-thread-friendly state updates -- a GUI front end's
/// window/tray icon reads these instead of parsing console output (the
/// console binary still prints everything it always did; this is a
/// parallel, decoupled channel, not a replacement for it).
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStatus {
    /// Everything loaded and the mic is live; not currently recording.
    Ready { mic_name: String, cleanup_enabled: bool },
    Recording,
    /// Hands-free is on and waiting for the user to start talking.
    Listening,
    Transcribing,
    Inserted(String),
    HeardNothing,
    HandsFreeOn,
    HandsFreeOff,
    Warning(String),
}

/// Events the blocking hotkey-listener thread hands off to the pipeline
/// worker (which does the actual, slower recording/transcribe/inject work
/// off the OS keyboard-hook thread), *and* the control surface a GUI uses
/// to inject the same events from a button/menu click instead of a key
/// press. There's deliberately one event type for both sources: hands-free
/// toggled from the tray menu should behave identically to hands-free
/// toggled from the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlEvent {
    PushToTalkPressed,
    PushToTalkReleased,
    /// Only the press edge of the hands-free toggle key matters -- it's a
    /// toggle, not a hold, so its release is never forwarded.
    HandsFreeTogglePressed,
    /// From the settings window's checkbox -- not reachable from a
    /// physical hotkey. Runtime-only: doesn't unload/reload the cleanup
    /// model (§2.4), just whether [`finish_utterance`] is allowed to use
    /// it for the next utterance onward.
    SetCleanupEnabled(bool),
    /// Ends [`Engine::run`]'s loop. Not reachable from a physical hotkey;
    /// only a GUI's "quit" action sends this.
    Quit,
}

/// Which interaction mode a session is running under -- determines what
/// ends it (a key release, or VAD-detected silence) and what happens
/// after it ends (nothing, or immediately start listening for the next
/// utterance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionMode {
    PushToTalk,
    HandsFree,
}

/// State for one recording session, from however it started (a hotkey
/// press, or a hands-free auto-restart) to however it ends (a hotkey
/// release, or VAD silence).
struct Session {
    mode: SessionMode,
    mark: Mark,
    /// Audio from just before `mark` (§2.1 pre-roll), captured once at
    /// press time. Session audio is conceptually `preroll ++
    /// read_since(mark)` -- see [`session_audio`] -- so every downstream
    /// consumer (VAD framing, window scheduling) just sees one continuous
    /// stream starting slightly before the key actually went down.
    preroll: Vec<f32>,
    /// How many full rolling windows have already been decoded.
    windows_decoded: usize,
    /// Merged transcript so far, stitched across window overlaps.
    committed: String,
    /// How many samples (from session start, i.e. including preroll) have
    /// already been fed to the VAD.
    vad_fed_samples: usize,
    /// Whether the endpointer has confirmed speech started yet this session.
    heard_speech: bool,
}

impl Session {
    /// Starts a new session of `mode`, capturing the mark and pre-roll
    /// snippet under one lock acquisition so `preroll` is exactly the
    /// audio ending right at the mark, with no gap or overlap.
    fn start(ring: &ring_buffer::SharedRingBuffer, mode: SessionMode) -> Self {
        let (mark, preroll) = {
            let guard = ring.lock().expect("ring buffer lock poisoned");
            let mark = guard.mark();
            let preroll = guard.read_last_duration(audio_input::TARGET_SAMPLE_RATE_HZ, PRE_ROLL_DURATION);
            (mark, preroll)
        };
        Self {
            mode,
            mark,
            preroll,
            windows_decoded: 0,
            committed: String::new(),
            vad_fed_samples: 0,
            heard_speech: false,
        }
    }
}

/// The full audio captured so far for `session`: pre-roll followed by
/// whatever's arrived since the session started.
fn session_audio(ring: &ring_buffer::SharedRingBuffer, session: &Session) -> Vec<f32> {
    let mut audio = session.preroll.clone();
    audio.extend(ring.lock().expect("ring buffer lock poisoned").read_since(session.mark));
    audio
}

/// ASR model path: first CLI arg, then `DICTATION_MODEL_PATH`, then the
/// default fetched by `crates/asr/README.md`'s instructions.
///
/// distil-small.en over small.en: measurably faster decode (fewer decoder
/// layers) on top of the `audio_ctx` fix (see `crates/asr/src/audio_ctx.rs`),
/// which is the change that actually mattered for real-time feel.
pub fn model_path() -> PathBuf {
    if let Some(arg) = std::env::args().nth(1) {
        return PathBuf::from(arg);
    }
    if let Ok(env_path) = std::env::var("DICTATION_MODEL_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("models/ggml-distil-small.en.bin")
}

/// VAD model path: `DICTATION_VAD_MODEL_PATH`, then the default fetched by
/// `crates/vad/README.md`'s instructions.
pub fn vad_model_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("DICTATION_VAD_MODEL_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("models/silero_vad.onnx")
}

/// Cleanup model path: `DICTATION_CLEANUP_MODEL_PATH`, then the default
/// fetched by `crates/cleanup/README.md`'s instructions.
pub fn cleanup_model_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("DICTATION_CLEANUP_MODEL_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

/// User dictionary path: `DICTATION_DICTIONARY_PATH`, then a plain-text
/// `dictionary.txt` in the working directory. See README.md for the
/// (one-term-per-line) format.
pub fn dictionary_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("DICTATION_DICTIONARY_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("dictionary.txt")
}

/// The loaded, running engine: models resident, mic live, hotkey hook
/// installed. Everything here needs to stay alive for the duration of
/// [`run`](Self::run) -- `_capture` in particular, since dropping it stops
/// mic capture.
pub struct Engine {
    ring: ring_buffer::SharedRingBuffer,
    transcriber: asr::Transcriber,
    vad: vad::SileroVad,
    cleanup: Option<cleanup::CleanupModel>,
    injector: inject::TextInjector,
    _capture: audio_input::AudioCapture,
    hotkey_config: MultiHotkeyConfig,
    tx: mpsc::Sender<ControlEvent>,
    rx: mpsc::Receiver<ControlEvent>,
    mic_name: String,
    /// Set while a push-to-talk session is open; the watchdog thread
    /// spawned in [`load`](Self::load) reads it to know when it should
    /// bother cross-checking real key state at all. See
    /// [`PTT_WATCHDOG_INTERVAL`].
    ptt_active: Arc<AtomicBool>,
}

impl Engine {
    /// Loads every model, opens the microphone, and installs the global
    /// hotkey hook. `print` is called with human-readable progress lines
    /// as loading proceeds -- the console binary passes `println!`
    /// directly; a GUI can pass a no-op or forward them to its own log.
    pub fn load(mut print: impl FnMut(&str)) -> Result<Self, EngineError> {
        let model_path = model_path();
        let vad_model_path = vad_model_path();

        print(&format!("ASR model: {}", model_path.display()));
        print(&format!("VAD model: {}", vad_model_path.display()));

        let mut asr_config = asr::AsrConfig::new(model_path);
        match asr::load_dictionary_file(dictionary_path()) {
            Ok(terms) => {
                asr_config.initial_prompt = asr::build_initial_prompt(&terms);
                match &asr_config.initial_prompt {
                    Some(_) => print(&format!("User dictionary: {} term(s) loaded", terms.len())),
                    None => print(&format!(
                        "User dictionary: {} is empty, no bias applied",
                        dictionary_path().display()
                    )),
                }
            }
            Err(_) => {
                // No dictionary file is the expected common case (§2.3's
                // custom vocab is an optional accuracy boost, not
                // required) -- just decode without a bias.
                print(&format!(
                    "User dictionary: none found at {} (optional; see README.md)",
                    dictionary_path().display()
                ));
            }
        }

        let transcriber = asr::Transcriber::load(asr_config)?;

        let vad = vad::SileroVad::load(vad_model_path)?;

        // §2.4: the cleanup pass is explicitly optional. Unlike ASR/VAD, a
        // missing model here doesn't stop the engine -- it just means
        // every utterance falls back to raw ASR text.
        let cleanup_model_path = cleanup_model_path();
        let cleanup = match cleanup::CleanupModel::load(&cleanup_model_path) {
            Ok(c) => {
                print(&format!("Cleanup model: {}", cleanup_model_path.display()));
                Some(c)
            }
            Err(e) => {
                print(&format!(
                    "Cleanup pass disabled (couldn't load {}): {e}",
                    cleanup_model_path.display()
                ));
                None
            }
        };

        let injector = inject::TextInjector::new()?;

        let ring = ring_buffer::shared_default_mono_16k();
        // Kept alive on `self` for the whole engine lifetime: dropping it
        // stops capture. This is the always-on ring buffer from §2.1 --
        // audio flows into it continuously, not just while a hotkey is
        // held.
        let capture = audio_input::AudioCapture::start(ring.clone())?;

        let mic_name = capture.device_name().to_string();
        print(&format!(
            "Mic is live: {mic_name} @ {}Hz/{}ch, resampled to {}Hz mono.",
            capture.device_sample_rate_hz(),
            capture.device_channels(),
            audio_input::TARGET_SAMPLE_RATE_HZ,
        ));
        print(
            "Buffer is in-memory only (last ~30s, continuously overwritten) -- \
             never written to disk.",
        );

        let hotkey_config = MultiHotkeyConfig::default();
        let (ptt_a, ptt_b) = hotkey_config.push_to_talk_keys;
        print(&format!(
            "Hold {ptt_a:?} + {ptt_b:?} together to dictate (push-to-talk); release either to insert at the cursor."
        ));
        print(&format!(
            "Tap {:?} to toggle hands-free mode (VAD silence commits each utterance).",
            hotkey_config.hands_free_toggle_key
        ));

        let (tx, rx) = mpsc::channel::<ControlEvent>();
        let hotkey_tx = tx.clone();
        std::thread::spawn(move || {
            let result = hotkey::listen_multi(hotkey_config, move |slot, event| {
                let mapped = match (slot, event) {
                    (HotkeySlot::PushToTalk, HotkeyEvent::Pressed) => Some(ControlEvent::PushToTalkPressed),
                    (HotkeySlot::PushToTalk, HotkeyEvent::Released) => Some(ControlEvent::PushToTalkReleased),
                    (HotkeySlot::HandsFreeToggle, HotkeyEvent::Pressed) => {
                        Some(ControlEvent::HandsFreeTogglePressed)
                    }
                    (HotkeySlot::HandsFreeToggle, HotkeyEvent::Released) => None,
                };
                // The receiver only goes away at process shutdown; nothing
                // useful to do with a send failure here.
                if let Some(mapped) = mapped {
                    let _ = hotkey_tx.send(mapped);
                }
            });
            if let Err(e) = result {
                eprintln!("error: hotkey listener stopped: {e}");
            }
        });

        // Watchdog: see PTT_WATCHDOG_INTERVAL's doc comment. Only ever
        // *forces* a release (by injecting the same ControlEvent a real
        // KeyRelease would produce) when it's actively told a session is
        // open and the OS says the chord's keys are no longer both
        // down -- never fabricates a press, never fires while
        // ptt_active is false, so it's a no-op unless the event stream
        // actually missed something.
        let ptt_active = Arc::new(AtomicBool::new(false));
        {
            let ptt_active = ptt_active.clone();
            let ptt_keys = hotkey_config.push_to_talk_keys;
            let watchdog_tx = tx.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(PTT_WATCHDOG_INTERVAL);
                if ptt_active.load(Ordering::Relaxed) && !hotkey::both_physically_down(ptt_keys.0, ptt_keys.1) {
                    let _ = watchdog_tx.send(ControlEvent::PushToTalkReleased);
                    // run()'s handler for that event clears this too once
                    // it actually processes the release; clearing it here
                    // as well just avoids re-sending every tick in the
                    // (very short) window before it does.
                    ptt_active.store(false, Ordering::Relaxed);
                }
            });
        }

        Ok(Self {
            ring,
            transcriber,
            vad,
            cleanup,
            injector,
            _capture: capture,
            hotkey_config,
            tx,
            rx,
            mic_name,
            ptt_active,
        })
    }

    pub fn hotkey_config(&self) -> MultiHotkeyConfig {
        self.hotkey_config
    }

    pub fn mic_name(&self) -> &str {
        &self.mic_name
    }

    pub fn cleanup_enabled(&self) -> bool {
        self.cleanup.is_some()
    }

    /// A sender for the exact same event stream the real hotkey listener
    /// feeds -- a GUI uses this to inject `HandsFreeTogglePressed` from a
    /// menu click or `Quit` to stop [`run`](Self::run), and both behave
    /// identically to their hotkey-triggered counterparts (there's no
    /// separate "UI-triggered" code path to keep in sync).
    pub fn control_sender(&self) -> mpsc::Sender<ControlEvent> {
        self.tx.clone()
    }

    /// Runs the pipeline until a [`ControlEvent::Quit`] arrives (or the
    /// hotkey thread dies and drops its sender). Blocks the calling
    /// thread -- run this on a background thread in a GUI app, since the
    /// GUI event loop needs the main thread on most platforms.
    pub fn run(mut self, status_tx: mpsc::Sender<PipelineStatus>) {
        let _ = status_tx.send(PipelineStatus::Ready {
            mic_name: self.mic_name.clone(),
            cleanup_enabled: self.cleanup.is_some(),
        });

        let window_policy = asr::WindowPolicy::default_16k();
        let mut endpointer = vad::Endpointer::new(vad::EndpointConfig::default());
        let mut session: Option<Session> = None;
        let mut hands_free_active = false;
        // Settings-window toggle (SetCleanupEnabled); starts true iff a
        // cleanup model actually loaded -- flipping it doesn't
        // unload/reload the model, just whether finish_utterance is
        // allowed to reach for it.
        let mut cleanup_runtime_enabled = self.cleanup.is_some();

        loop {
            let event = if session.is_some() {
                match self.rx.recv_timeout(POLL_INTERVAL) {
                    Ok(event) => Some(event),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match self.rx.recv() {
                    Ok(event) => Some(event),
                    Err(_disconnected) => break,
                }
            };

            match event {
                Some(ControlEvent::Quit) => break,
                Some(ControlEvent::PushToTalkPressed) => {
                    if session.is_some() {
                        continue; // a session (of either mode) is already running; ignore
                    }
                    self.vad.reset_state();
                    endpointer.reset();
                    session = Some(Session::start(&self.ring, SessionMode::PushToTalk));
                    self.ptt_active.store(true, Ordering::Relaxed);
                    println!("Recording... (with {PRE_ROLL_DURATION:?} pre-roll)");
                    let _ = status_tx.send(PipelineStatus::Recording);
                }
                Some(ControlEvent::PushToTalkReleased) => {
                    let Some(mut s) = session.take() else {
                        continue; // release without a matching press; ignore
                    };
                    if s.mode != SessionMode::PushToTalk {
                        // A hands-free session is running; a stray
                        // push-to-talk release (its key can't release
                        // without a matching press, so this shouldn't
                        // happen in practice) must not steal and end it.
                        session = Some(s);
                        continue;
                    }
                    // Real key release or the watchdog's synthesized one
                    // (see PTT_WATCHDOG_INTERVAL) -- either way, the
                    // session is ending now.
                    self.ptt_active.store(false, Ordering::Relaxed);
                    self.commit_utterance(&window_policy, &mut s, &status_tx, cleanup_runtime_enabled);
                }
                Some(ControlEvent::SetCleanupEnabled(enabled)) => {
                    cleanup_runtime_enabled = enabled && self.cleanup.is_some();
                    println!(
                        "Cleanup pass: {}",
                        if cleanup_runtime_enabled { "ON" } else { "OFF" }
                    );
                }
                Some(ControlEvent::HandsFreeTogglePressed) => {
                    hands_free_active = !hands_free_active;
                    if hands_free_active {
                        println!("Hands-free mode: ON (VAD silence commits each utterance)");
                        let _ = status_tx.send(PipelineStatus::HandsFreeOn);
                        if session.is_none() {
                            self.vad.reset_state();
                            endpointer.reset();
                            session = Some(Session::start(&self.ring, SessionMode::HandsFree));
                            println!("Listening... (with {PRE_ROLL_DURATION:?} pre-roll)");
                            let _ = status_tx.send(PipelineStatus::Listening);
                        }
                    } else {
                        println!("Hands-free mode: OFF");
                        let _ = status_tx.send(PipelineStatus::HandsFreeOff);
                        if let Some(mut s) = session.take() {
                            if s.mode == SessionMode::HandsFree {
                                self.commit_utterance(&window_policy, &mut s, &status_tx, cleanup_runtime_enabled);
                            } else {
                                session = Some(s); // leave an unrelated push-to-talk session alone
                            }
                        }
                    }
                }
                None => {
                    // Poll timeout: feed any newly-arrived audio to the
                    // VAD, and decode a new window if one's ready and
                    // we've actually heard speech (§2.2: trim silence
                    // rather than spending decode time on it).
                    let Some(s) = session.as_mut() else {
                        continue;
                    };
                    let audio = session_audio(&self.ring, s);
                    let vad_event = feed_vad(&mut self.vad, &mut endpointer, &audio, s);
                    if s.heard_speech {
                        if let Some((start, end)) = window_policy.next_window(audio.len(), s.windows_decoded) {
                            decode_window(&self.transcriber, &audio[start..end], &mut s.committed);
                            s.windows_decoded += 1;
                        }
                    }

                    // §2.2: "hands-free (VAD silence = commit)".
                    // Push-to-talk sessions ignore VAD end-of-speech
                    // entirely -- release is still what commits them.
                    let hands_free_speech_ended =
                        s.mode == SessionMode::HandsFree && vad_event == Some(vad::EndpointEvent::SpeechEnd);
                    if hands_free_speech_ended {
                        let mut finished = session.take().expect("checked Some above");
                        self.commit_utterance(&window_policy, &mut finished, &status_tx, cleanup_runtime_enabled);
                        // Hands-free keeps listening: immediately start
                        // the next utterance's session rather than
                        // waiting for another toggle press.
                        self.vad.reset_state();
                        endpointer.reset();
                        session = Some(Session::start(&self.ring, SessionMode::HandsFree));
                        println!("Listening...");
                        let _ = status_tx.send(PipelineStatus::Listening);
                    }
                }
            }
        }
    }

    /// Decodes the trailing partial window and inserts the final text
    /// (§2.4: "pick one string, insert once"). Shared by every way a
    /// session can end: push-to-talk release, hands-free toggle-off, and
    /// hands-free VAD-detected silence.
    fn commit_utterance(
        &mut self,
        window_policy: &asr::WindowPolicy,
        session: &mut Session,
        status_tx: &mpsc::Sender<PipelineStatus>,
        cleanup_runtime_enabled: bool,
    ) {
        let _ = status_tx.send(PipelineStatus::Transcribing);
        let audio = session_audio(&self.ring, session);
        if let Some((start, end)) = window_policy.final_window(audio.len(), session.windows_decoded) {
            decode_window(&self.transcriber, &audio[start..end], &mut session.committed);
        }
        // Settings-window toggle: skip reaching for the model at all
        // when it's off, same as if none had ever loaded.
        let cleanup = cleanup_runtime_enabled.then_some(self.cleanup.as_ref()).flatten();
        finish_utterance(&session.committed, cleanup, &mut self.injector, status_tx);
    }
}

/// Feeds newly-arrived audio to the VAD frame by frame, updating the
/// endpointer and `session.heard_speech`. Returns the last endpoint event
/// observed during this call, if any -- callers that care about
/// VAD-triggered commit (hands-free) act on `SpeechEnd`; push-to-talk
/// sessions just ignore the return value.
fn feed_vad(
    vad: &mut vad::SileroVad,
    endpointer: &mut vad::Endpointer,
    audio: &[f32],
    session: &mut Session,
) -> Option<vad::EndpointEvent> {
    let mut last_event = None;
    while let Some((start, end)) = vad::next_frame_range(session.vad_fed_samples, audio.len()) {
        match vad.process_frame(&audio[start..end]) {
            Ok(probability) => {
                if let Some(event) = endpointer.push_probability(probability) {
                    if event == vad::EndpointEvent::SpeechStart && !session.heard_speech {
                        println!("(speech detected)");
                    }
                    if event == vad::EndpointEvent::SpeechStart {
                        session.heard_speech = true;
                    }
                    last_event = Some(event);
                }
            }
            Err(e) => eprintln!("warning: VAD frame failed: {e}"),
        }
        session.vad_fed_samples = end;
    }
    last_event
}

fn decode_window(transcriber: &asr::Transcriber, audio: &[f32], committed: &mut String) {
    match transcriber.transcribe(audio) {
        Ok(text) if !text.trim().is_empty() => {
            *committed = asr::merge_overlap(committed, &text);
            println!("... {committed}");
        }
        Ok(_) => {}
        Err(e) => eprintln!("warning: transcription failed: {e}"),
    }
}

/// Picks exactly one final string and inserts it once (§2.4: "Do not
/// insert raw text and then patch it afterward"). If a cleanup model is
/// configured, this is the one place its deadlined output is raced and
/// decided -- never revisited afterward, whichever way it comes out.
fn finish_utterance(
    committed: &str,
    cleanup: Option<&cleanup::CleanupModel>,
    injector: &mut inject::TextInjector,
    status_tx: &mpsc::Sender<PipelineStatus>,
) {
    if committed.trim().is_empty() {
        println!("(heard nothing)");
        let _ = status_tx.send(PipelineStatus::HeardNothing);
        return;
    }

    let cleaned = cleanup.and_then(|c| c.clean(committed));
    let final_text = match &cleaned {
        Some(text) if !text.trim().is_empty() => {
            println!("-> {text}");
            text.as_str()
        }
        _ => {
            println!("-> {committed}");
            committed
        }
    };

    if let Err(e) = injector.inject(final_text) {
        eprintln!("warning: couldn't insert text: {e}");
        let _ = status_tx.send(PipelineStatus::Warning(format!("couldn't insert text: {e}")));
        return;
    }
    let _ = status_tx.send(PipelineStatus::Inserted(final_text.to_string()));
}
