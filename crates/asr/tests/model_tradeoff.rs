//! Speed *and* accuracy for each candidate model, side by side, through
//! the real `Transcriber` on real speech.
//!
//! This is the benchmark that should have existed from the start.
//! `model_benchmark.rs` compares wall-clock only, on a synthetic sine --
//! and optimizing against exactly that is what let `audio_ctx` scoping
//! ship repetition loops while looking like an 11.7x win. A dictation
//! model choice is a two-axis decision and a one-axis benchmark can only
//! mislead.
//!
//! Uses `AsrConfig::new` (not raw whisper params) so what's measured is
//! what production actually runs, `audio_ctx` floor included.
//!
//! ASR_MODELS=<comma-separated .bin paths> ASR_SPEECH_WAV=<path to a 16k mono wav> \
//!   cargo test --release -p asr --test model_tradeoff -- --ignored --nocapture

use std::time::Instant;

use asr::{AsrConfig, Transcriber};

fn read_wav_16k_mono(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("fixture wav should exist");
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"data" {
            let end = (body + size).min(bytes.len());
            return bytes[body..end]
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                .collect();
        }
        pos = body + size + (size & 1);
    }
    panic!("no data chunk in {path}");
}

#[test]
#[ignore]
fn compare_models_on_speed_and_transcript() {
    let (Ok(models), Ok(wav_path)) = (std::env::var("ASR_MODELS"), std::env::var("ASR_SPEECH_WAV")) else {
        eprintln!("set ASR_MODELS (comma-separated) and ASR_SPEECH_WAV");
        return;
    };

    let full = read_wav_16k_mono(&wav_path);
    let seconds = full.len() as f64 / 16_000.0;
    println!("fixture: {seconds:.2}s of real speech\n");

    // Two lengths that matter in production: a whole short utterance, and
    // the kind of short trailing window `WindowPolicy::final_window`
    // hands over at hotkey release.
    let tail: Vec<f32> = full.iter().copied().take(16_000 / 2).collect();

    for model_path in models.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let name = model_path.rsplit(['/', '\\']).next().unwrap_or(model_path);
        let load_start = Instant::now();
        let transcriber = match Transcriber::load(AsrConfig::new(model_path)) {
            Ok(t) => t,
            Err(e) => {
                println!("{name}: FAILED to load: {e}\n");
                continue;
            }
        };
        let load_ms = load_start.elapsed().as_millis();

        for (label, audio) in [("full utterance", &full), ("0.5s tail", &tail)] {
            let _ = transcriber.transcribe(audio); // warm-up
            let start = Instant::now();
            let text = transcriber.transcribe(audio).unwrap_or_default();
            let ms = start.elapsed().as_millis();
            let shown: String = text.trim().chars().take(90).collect();
            println!("{name:<26} {label:<15} {ms:>6}ms  \"{shown}\"");
        }
        println!("{name:<26} load {load_ms}ms\n");
    }

    println!(
        "Pick on both columns. A model that is fast and wrong is not a\n\
         speed win -- it just moves the cost to the user retyping."
    );
}
