//! Resident daemon: wires the ring buffer, mic capture, push-to-talk
//! hotkey, batch Whisper transcription, and clipboard text injection
//! together into the v0 demo pipeline from `dictation-architecture.md`.
//!
//! v0 shape (§4): hold hotkey -> record, release -> batch-transcribe the
//! whole utterance -> insert at cursor. No VAD, no streaming, no cleanup
//! pass yet -- those land in v1/v2 on top of this same skeleton.

use std::path::PathBuf;
use std::sync::mpsc;

use hotkey::{HotkeyConfig, HotkeyEvent};
use ring_buffer::Mark;

/// Events the blocking hotkey-listener thread hands off to the pipeline
/// worker, which does the actual (slower) recording/transcribe/inject
/// work off the OS keyboard-hook thread.
enum PipelineEvent {
    Pressed,
    Released,
}

fn main() {
    let model_path = model_path();

    println!("Local Dictation Engine -- v0 demo");
    println!("Model: {}", model_path.display());

    let transcriber = match asr::Transcriber::load(asr::AsrConfig::new(model_path)) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: couldn't load ASR model: {e}");
            eprintln!("See crates/asr/README.md for how to fetch a model file.");
            std::process::exit(1);
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

    run_pipeline(rx, &ring, &transcriber, &mut injector);
}

/// The main worker loop: turns hotkey press/release pairs into recorded
/// audio, transcribed text, and inserted text. Runs on its own thread (via
/// the channel), separate from the OS-level keyboard hook, so a slow
/// transcription never risks the hook watchdog thinking the hook is stuck.
fn run_pipeline(
    rx: mpsc::Receiver<PipelineEvent>,
    ring: &ring_buffer::SharedRingBuffer,
    transcriber: &asr::Transcriber,
    injector: &mut inject::TextInjector,
) {
    let mut session_start: Option<Mark> = None;

    for event in rx {
        match event {
            PipelineEvent::Pressed => {
                let mark = ring.lock().expect("ring buffer lock poisoned").mark();
                session_start = Some(mark);
                println!("Recording...");
            }
            PipelineEvent::Released => {
                let Some(mark) = session_start.take() else {
                    continue; // release without a matching press; ignore
                };
                let audio = ring.lock().expect("ring buffer lock poisoned").read_since(mark);
                if audio.is_empty() {
                    println!("(no audio captured)");
                    continue;
                }

                match transcriber.transcribe(&audio) {
                    Ok(text) if text.trim().is_empty() => {
                        println!("(heard nothing)");
                    }
                    Ok(text) => {
                        println!("-> {text}");
                        if let Err(e) = injector.inject(&text) {
                            eprintln!("warning: couldn't insert text: {e}");
                        }
                    }
                    Err(e) => {
                        eprintln!("warning: transcription failed: {e}");
                    }
                }
            }
        }
    }
}

/// Model path: first CLI arg, then `DICTATION_MODEL_PATH`, then the
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
