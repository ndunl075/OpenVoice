//! Does scoping `audio_ctx` down actually preserve the transcript?
//!
//! Every other benchmark in this directory measures *time*. That turned
//! out to be the whole problem: `audio_ctx` scoping was introduced and
//! tuned purely against wall-clock numbers, on a synthetic sine wave,
//! and nobody ever looked at the text coming out. In production it
//! produced repetition loops -- "Hey, what's going on? Hey, what's going
//! on? ... What? What? What?" -- from a two-second utterance.
//!
//! `dictation-architecture.md` §5 names this exact failure mode: "If you
//! win on milliseconds and lose on output quality, you have built a
//! worse product with a better benchmark."
//!
//! So this test asserts on **text**, against real speech with known
//! ground truth rather than a sine wave (a sine isn't speech, so whisper
//! hallucinates on it no matter what -- which is precisely why timing
//! runs against it were misleading).
//!
//! Generate the fixtures first (Windows, no extra dependencies):
//!
//! ```powershell
//! Add-Type -AssemblyName System.Speech
//! $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
//! $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
//! $s.SetOutputToWaveFile("speech_hey.wav", $fmt)
//! $s.Speak("Hey, what's going on")
//! $s.Dispose()
//! ```
//!
//! Then:
//! ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin ASR_SPEECH_WAV=<path to speech_hey.wav> \
//!   cargo test --release -p asr --test audio_ctx_quality -- --ignored --nocapture

use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Minimal 16-bit PCM mono WAV reader -- enough for the fixtures above,
/// deliberately not a general-purpose decoder (no extra dependency for a
/// test helper).
fn read_wav_16k_mono(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("fixture wav should exist");
    // Walk RIFF chunks to find `data` rather than assuming a fixed 44-byte
    // header -- SAPI writes extra chunks before it.
    let mut pos = 12; // skip "RIFF" + size + "WAVE"
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
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    panic!("no data chunk in {path}");
}

/// Cheap repetition detector: the failure mode is a phrase repeating over
/// and over, which shows up as very low unique-word ratio.
fn repetition_ratio(text: &str) -> f64 {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return 1.0;
    }
    let unique: std::collections::HashSet<&String> = words.iter().collect();
    unique.len() as f64 / words.len() as f64
}

#[test]
#[ignore]
fn sweep_audio_ctx_and_show_the_actual_transcript() {
    let (Ok(model_path), Ok(wav_path)) = (
        std::env::var("ASR_BENCH_MODEL_A"),
        std::env::var("ASR_SPEECH_WAV"),
    ) else {
        eprintln!("set ASR_BENCH_MODEL_A and ASR_SPEECH_WAV to run this");
        return;
    };

    let mut audio = read_wav_16k_mono(&wav_path);
    // Optional truncation: the trailing window at hotkey release is short
    // by construction, and short clips are exactly where a too-small
    // audio_ctx degenerates worst -- so the safe floor has to be
    // validated against those, not just against a full utterance.
    if let Ok(secs) = std::env::var("ASR_TRUNCATE_SECS") {
        if let Ok(secs) = secs.parse::<f64>() {
            audio.truncate((secs * 16_000.0) as usize);
        }
    }
    let seconds = audio.len() as f64 / 16_000.0;
    println!("speech fixture: {seconds:.2}s ({} samples)\n", audio.len());

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("model should load");

    // 0 means whisper's default (full 30s context). The rest span what
    // audio_ctx_for would produce for clips this length, up through
    // clearly-safe values.
    for audio_ctx in [0, 64, 128, 256, 512, 768, 1000, 1500] {
        let mut state = ctx.create_state().expect("state");
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(8);
        params.set_language(Some("en"));
        params.set_no_context(true);
        params.set_no_timestamps(true);
        params.set_suppress_blank(true);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0);
        params.set_audio_ctx(audio_ctx);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        let start = Instant::now();
        state.full(params, &audio).expect("decode");
        let elapsed = start.elapsed();

        let mut text = String::new();
        for i in 0..state.full_n_segments() {
            if let Some(seg) = state.get_segment(i) {
                text.push_str(&seg.to_str_lossy().unwrap_or_default());
            }
        }
        let text = text.trim();
        let ratio = repetition_ratio(text);
        let label = if ratio < 0.35 { "REPETITION LOOP" } else { "ok" };
        let shown: String = text.chars().take(110).collect();
        println!(
            "audio_ctx {audio_ctx:>4}: {elapsed:>8.2?} | unique-word ratio {ratio:.2} {label}\n            \"{shown}\"\n"
        );
    }
}
