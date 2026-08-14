//! Resident daemon: wires the ring buffer, mic capture, push-to-talk
//! hotkey, VAD-gated streaming Whisper transcription, an optional
//! deadlined cleanup pass, and clipboard text injection together per
//! `dictation-architecture.md`.
//!
//! v1 shape (§4): still push-to-talk (hands-free VAD-commit is v2), but
//! decoding happens continuously *while the key is held* instead of
//! waiting for release. Rolling 3s/0.5s-overlap windows (§2.3) get
//! transcribed as soon as they're ready, stitched together with
//! `asr::merge_overlap`; at release, only the trailing partial window is
//! still outstanding. Silero VAD (§2.2) gates window decoding on whether
//! the endpointer has actually confirmed speech yet, so holding the
//! hotkey while thinking doesn't burn decode time on silence.
//!
//! v2 additions: once the utterance's raw text is assembled, an optional
//! cleanup LLM pass (§2.4) races a 120ms deadline to punctuate/declutter
//! it before `finish_utterance` picks exactly one final string to insert.
//! Every session also includes ~500ms of pre-roll (§2.1) -- audio from
//! just before the hotkey went down, pulled from the always-on ring
//! buffer -- so starting to talk a beat before pressing the key doesn't
//! lose the first word. And per §2.2/§4, both interaction modes are
//! supported: push-to-talk (hold to record, release commits) and
//! hands-free (tap a separate toggle key to start listening; VAD silence
//! commits each utterance and immediately starts listening for the next
//! one, until the toggle key is tapped again to stop).

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use hotkey::{HotkeyEvent, HotkeySlot, MultiHotkeyConfig};
use ring_buffer::Mark;

/// Events the blocking hotkey-listener thread hands off to the pipeline
/// worker, which does the actual (slower) recording/transcribe/inject
/// work off the OS keyboard-hook thread.
enum PipelineEvent {
    PushToTalkPressed,
    PushToTalkReleased,
    /// Only the press edge of the hands-free toggle key matters -- it's a
    /// toggle, not a hold, so its release is never forwarded.
    HandsFreeTogglePressed,
}

/// §2.1: "capture the ~500 ms *before* the hotkey press. Solves the
/// universal 'I started talking a beat too early and lost my first word'
/// problem." The ring buffer's always-on capture is what makes this
/// possible at all -- there's no re-opening a device to catch up on.
const PRE_ROLL_DURATION: Duration = Duration::from_millis(500);

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

fn main() {
    let model_path = model_path();
    let vad_model_path = vad_model_path();

    println!("Local Dictation Engine -- v1 (streaming + VAD)");
    println!("ASR model: {}", model_path.display());
    println!("VAD model: {}", vad_model_path.display());

    let mut asr_config = asr::AsrConfig::new(model_path);
    match asr::load_dictionary_file(dictionary_path()) {
        Ok(terms) => {
            asr_config.initial_prompt = asr::build_initial_prompt(&terms);
            match &asr_config.initial_prompt {
                Some(_) => println!("User dictionary: {} term(s) loaded", terms.len()),
                None => println!("User dictionary: {} is empty, no bias applied", dictionary_path().display()),
            }
        }
        Err(_) => {
            // No dictionary file is the expected common case (§2.3's
            // custom vocab is an optional accuracy boost, not required) --
            // just decode without a bias, same as before this existed.
            println!(
                "User dictionary: none found at {} (optional; see README.md)",
                dictionary_path().display()
            );
        }
    }

    let transcriber = match asr::Transcriber::load(asr_config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: couldn't load ASR model: {e}");
            eprintln!("See crates/asr/README.md for how to fetch a model file.");
            std::process::exit(1);
        }
    };

    let mut vad = match vad::SileroVad::load(vad_model_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: couldn't load VAD model: {e}");
            eprintln!("See crates/vad/README.md for how to fetch a model file.");
            std::process::exit(1);
        }
    };

    // §2.4: the cleanup pass is explicitly optional. Unlike ASR/VAD, a
    // missing model here doesn't stop the daemon -- it just means every
    // utterance falls back to raw ASR text, same as before this feature
    // existed.
    let cleanup_model_path = cleanup_model_path();
    let cleanup = match cleanup::CleanupModel::load(&cleanup_model_path) {
        Ok(c) => {
            println!("Cleanup model: {}", cleanup_model_path.display());
            Some(c)
        }
        Err(e) => {
            println!(
                "Cleanup pass disabled (couldn't load {}): {e}",
                cleanup_model_path.display()
            );
            println!("See crates/cleanup/README.md for how to fetch a model file.");
            None
        }
    };

    let mut injector = match inject::TextInjector::new() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: couldn't initialize text injection: {e}");
            std::process::exit(1);
        }
    };

    let ring = ring_buffer::shared_default_mono_16k();

    // Kept alive for the whole process lifetime: dropping it stops capture.
    // This is the always-on ring buffer from §2.1 -- audio flows into it
    // continuously, not just while the hotkey is held.
    let capture = match audio_input::AudioCapture::start(ring.clone()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: couldn't start microphone capture: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Mic is live: {} @ {}Hz/{}ch, resampled to {}Hz mono.",
        capture.device_name(),
        capture.device_sample_rate_hz(),
        capture.device_channels(),
        audio_input::TARGET_SAMPLE_RATE_HZ,
    );
    println!(
        "Buffer is in-memory only (last ~30s, continuously overwritten) -- \
         never written to disk. See README.md for details."
    );

    let hotkey_config = MultiHotkeyConfig::default();
    println!(
        "Hold {:?} to dictate (push-to-talk); release to insert at the cursor.",
        hotkey_config.push_to_talk_key
    );
    println!(
        "Tap {:?} to toggle hands-free mode (VAD silence commits each utterance). Ctrl+C to quit.",
        hotkey_config.hands_free_toggle_key
    );

    let (tx, rx) = mpsc::channel::<PipelineEvent>();
    std::thread::spawn(move || {
        let result = hotkey::listen_multi(hotkey_config, move |slot, event| {
            let mapped = match (slot, event) {
                (HotkeySlot::PushToTalk, HotkeyEvent::Pressed) => Some(PipelineEvent::PushToTalkPressed),
                (HotkeySlot::PushToTalk, HotkeyEvent::Released) => Some(PipelineEvent::PushToTalkReleased),
                (HotkeySlot::HandsFreeToggle, HotkeyEvent::Pressed) => Some(PipelineEvent::HandsFreeTogglePressed),
                (HotkeySlot::HandsFreeToggle, HotkeyEvent::Released) => None,
            };
            // The receiver only goes away at process shutdown; nothing
            // useful to do with a send failure here.
            if let Some(mapped) = mapped {
                let _ = tx.send(mapped);
            }
        });
        if let Err(e) = result {
            eprintln!("error: hotkey listener stopped: {e}");
            std::process::exit(1);
        }
    });

    run_pipeline(rx, &ring, &transcriber, &mut vad, cleanup.as_ref(), &mut injector);
}

/// How often the worker checks for newly-ready VAD frames / decode
/// windows while the hotkey is held. Small enough that a window becomes
/// available for decode close to the moment its last sample lands, large
/// enough not to spin the loop pointlessly.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The main worker loop. Runs on its own thread (via the channel),
/// separate from the OS-level keyboard hook, so a transcription can never
/// risk the hook watchdog thinking the hook is stuck. While idle it just
/// blocks on the next hotkey event; while recording it polls at
/// [`POLL_INTERVAL`] so it can decode windows as they become ready without
/// waiting for release.
fn run_pipeline(
    rx: mpsc::Receiver<PipelineEvent>,
    ring: &ring_buffer::SharedRingBuffer,
    transcriber: &asr::Transcriber,
    vad: &mut vad::SileroVad,
    cleanup: Option<&cleanup::CleanupModel>,
    injector: &mut inject::TextInjector,
) {
    let window_policy = asr::WindowPolicy::default_16k();
    let mut endpointer = vad::Endpointer::new(vad::EndpointConfig::default());
    let mut session: Option<Session> = None;
    let mut hands_free_active = false;

    loop {
        let event = if session.is_some() {
            match rx.recv_timeout(POLL_INTERVAL) {
                Ok(event) => Some(event),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(event) => Some(event),
                Err(_disconnected) => break,
            }
        };

        match event {
            Some(PipelineEvent::PushToTalkPressed) => {
                if session.is_some() {
                    continue; // a session (of either mode) is already running; ignore
                }
                vad.reset_state();
                endpointer.reset();
                session = Some(Session::start(ring, SessionMode::PushToTalk));
                println!("Recording... (with {PRE_ROLL_DURATION:?} pre-roll)");
            }
            Some(PipelineEvent::PushToTalkReleased) => {
                let Some(mut s) = session.take() else {
                    continue; // release without a matching press; ignore
                };
                if s.mode != SessionMode::PushToTalk {
                    // A hands-free session is running; a stray push-to-talk
                    // release (its key can't release without a matching
                    // press, so this shouldn't happen in practice) must not
                    // steal and end it.
                    session = Some(s);
                    continue;
                }
                commit_utterance(ring, transcriber, &window_policy, cleanup, injector, &mut s);
            }
            Some(PipelineEvent::HandsFreeTogglePressed) => {
                hands_free_active = !hands_free_active;
                if hands_free_active {
                    println!("Hands-free mode: ON (VAD silence commits each utterance)");
                    if session.is_none() {
                        vad.reset_state();
                        endpointer.reset();
                        session = Some(Session::start(ring, SessionMode::HandsFree));
                        println!("Listening... (with {PRE_ROLL_DURATION:?} pre-roll)");
                    }
                } else {
                    println!("Hands-free mode: OFF");
                    if let Some(mut s) = session.take() {
                        if s.mode == SessionMode::HandsFree {
                            commit_utterance(ring, transcriber, &window_policy, cleanup, injector, &mut s);
                        } else {
                            session = Some(s); // leave an unrelated push-to-talk session alone
                        }
                    }
                }
            }
            None => {
                // Poll timeout: feed any newly-arrived audio to the VAD,
                // and decode a new window if one's ready and we've
                // actually heard speech (§2.2: trim silence rather than
                // spending decode time on it).
                let Some(s) = session.as_mut() else {
                    continue;
                };
                let audio = session_audio(ring, s);
                let vad_event = feed_vad(vad, &mut endpointer, &audio, s);
                if s.heard_speech {
                    if let Some((start, end)) = window_policy.next_window(audio.len(), s.windows_decoded) {
                        decode_window(transcriber, &audio[start..end], &mut s.committed);
                        s.windows_decoded += 1;
                    }
                }

                // §2.2: "hands-free (VAD silence = commit)". Push-to-talk
                // sessions ignore VAD end-of-speech entirely -- release is
                // still what commits them.
                let hands_free_speech_ended =
                    s.mode == SessionMode::HandsFree && vad_event == Some(vad::EndpointEvent::SpeechEnd);
                if hands_free_speech_ended {
                    let mut finished = session.take().expect("checked Some above");
                    commit_utterance(ring, transcriber, &window_policy, cleanup, injector, &mut finished);
                    // Hands-free keeps listening: immediately start the next
                    // utterance's session rather than waiting for another
                    // toggle press.
                    vad.reset_state();
                    endpointer.reset();
                    session = Some(Session::start(ring, SessionMode::HandsFree));
                    println!("Listening...");
                }
            }
        }
    }
}

/// Decodes the trailing partial window and inserts the final text (§2.4:
/// "pick one string, insert once"). Shared by every way a session can end:
/// push-to-talk release, hands-free toggle-off, and hands-free
/// VAD-detected silence.
fn commit_utterance(
    ring: &ring_buffer::SharedRingBuffer,
    transcriber: &asr::Transcriber,
    window_policy: &asr::WindowPolicy,
    cleanup: Option<&cleanup::CleanupModel>,
    injector: &mut inject::TextInjector,
    session: &mut Session,
) {
    let audio = session_audio(ring, session);
    if let Some((start, end)) = window_policy.final_window(audio.len(), session.windows_decoded) {
        decode_window(transcriber, &audio[start..end], &mut session.committed);
    }
    finish_utterance(&session.committed, cleanup, injector);
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
fn finish_utterance(committed: &str, cleanup: Option<&cleanup::CleanupModel>, injector: &mut inject::TextInjector) {
    if committed.trim().is_empty() {
        println!("(heard nothing)");
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
    }
}

/// ASR model path: first CLI arg, then `DICTATION_MODEL_PATH`, then the
/// default fetched by `crates/asr/README.md`'s instructions.
fn model_path() -> PathBuf {
    if let Some(arg) = std::env::args().nth(1) {
        return PathBuf::from(arg);
    }
    if let Ok(env_path) = std::env::var("DICTATION_MODEL_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("models/ggml-small.en-q5_1.bin")
}

/// VAD model path: `DICTATION_VAD_MODEL_PATH`, then the default fetched by
/// `crates/vad/README.md`'s instructions.
fn vad_model_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("DICTATION_VAD_MODEL_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("models/silero_vad.onnx")
}

/// Cleanup model path: `DICTATION_CLEANUP_MODEL_PATH`, then the default
/// fetched by `crates/cleanup/README.md`'s instructions.
fn cleanup_model_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("DICTATION_CLEANUP_MODEL_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

/// User dictionary path: `DICTATION_DICTIONARY_PATH`, then a plain-text
/// `dictionary.txt` in the working directory. See README.md for the
/// (one-term-per-line) format.
fn dictionary_path() -> PathBuf {
    if let Ok(env_path) = std::env::var("DICTATION_DICTIONARY_PATH") {
        return PathBuf::from(env_path);
    }
    PathBuf::from("dictionary.txt")
}
