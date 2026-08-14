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
//! v2 addition: once the utterance's raw text is assembled, an optional
//! cleanup LLM pass (§2.4) races a 120ms deadline to punctuate/declutter
//! it before `finish_utterance` picks exactly one final string to insert.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use hotkey::{HotkeyConfig, HotkeyEvent};
use ring_buffer::Mark;

/// Events the blocking hotkey-listener thread hands off to the pipeline
/// worker, which does the actual (slower) recording/transcribe/inject
/// work off the OS keyboard-hook thread.
enum PipelineEvent {
    Pressed,
    Released,
}

/// State for one hotkey-down-to-up recording session.
struct Session {
    mark: Mark,
    /// How many full rolling windows have already been decoded.
    windows_decoded: usize,
    /// Merged transcript so far, stitched across window overlaps.
    committed: String,
    /// How many samples (from session start) have already been fed to the VAD.
    vad_fed_samples: usize,
    /// Whether the endpointer has confirmed speech started yet this session.
    heard_speech: bool,
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

    let hotkey_config = HotkeyConfig::default();
    println!(
        "Hold {:?} to dictate; release to insert at the cursor. Ctrl+C to quit.",
        hotkey_config.key
    );

    let (tx, rx) = mpsc::channel::<PipelineEvent>();
    std::thread::spawn(move || {
        let result = hotkey::listen_push_to_talk(hotkey_config, move |event| {
            let mapped = match event {
                HotkeyEvent::Pressed => PipelineEvent::Pressed,
                HotkeyEvent::Released => PipelineEvent::Released,
            };
            // The receiver only goes away at process shutdown; nothing
            // useful to do with a send failure here.
            let _ = tx.send(mapped);
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
            Some(PipelineEvent::Pressed) => {
                let mark = ring.lock().expect("ring buffer lock poisoned").mark();
                vad.reset_state();
                endpointer.reset();
                session = Some(Session {
                    mark,
                    windows_decoded: 0,
                    committed: String::new(),
                    vad_fed_samples: 0,
                    heard_speech: false,
                });
                println!("Recording...");
            }
            Some(PipelineEvent::Released) => {
                let Some(mut s) = session.take() else {
                    continue; // release without a matching press; ignore
                };
                let audio = ring.lock().expect("ring buffer lock poisoned").read_since(s.mark);
                if let Some((start, end)) = window_policy.final_window(audio.len(), s.windows_decoded) {
                    decode_window(transcriber, &audio[start..end], &mut s.committed);
                }
                finish_utterance(&s.committed, cleanup, injector);
            }
            None => {
                // Poll timeout: feed any newly-arrived audio to the VAD,
                // and decode a new window if one's ready and we've
                // actually heard speech (§2.2: trim silence rather than
                // spending decode time on it).
                if let Some(s) = session.as_mut() {
                    let audio = ring.lock().expect("ring buffer lock poisoned").read_since(s.mark);
                    feed_vad(vad, &mut endpointer, &audio, s);
                    if s.heard_speech {
                        if let Some((start, end)) = window_policy.next_window(audio.len(), s.windows_decoded) {
                            decode_window(transcriber, &audio[start..end], &mut s.committed);
                            s.windows_decoded += 1;
                        }
                    }
                }
            }
        }
    }
}

fn feed_vad(vad: &mut vad::SileroVad, endpointer: &mut vad::Endpointer, audio: &[f32], session: &mut Session) {
    while let Some((start, end)) = vad::next_frame_range(session.vad_fed_samples, audio.len()) {
        match vad.process_frame(&audio[start..end]) {
            Ok(probability) => {
                if let Some(vad::EndpointEvent::SpeechStart) = endpointer.push_probability(probability) {
                    if !session.heard_speech {
                        println!("(speech detected)");
                    }
                    session.heard_speech = true;
                }
            }
            Err(e) => eprintln!("warning: VAD frame failed: {e}"),
        }
        session.vad_fed_samples = end;
    }
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
