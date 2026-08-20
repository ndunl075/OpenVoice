//! Probe: is a *small* `audio_ctx` pathologically slow?
//!
//! `commit_latency.rs` turned up a result that makes no sense on its
//! face: a 0.25s clip decoding in ~4.3s, while a 1.0s clip -- four times
//! the audio -- decodes in ~0.9s. Shorter input, 5x the time. That was
//! first assumed to be whisper.cpp's temperature-fallback retry ladder,
//! but sweeping `temperature_inc` (including 0.0, which disables the
//! ladder entirely) reproduced the spike unchanged, so it's something
//! else.
//!
//! Next hypothesis: `audio_ctx` itself. `audio_ctx_for` scopes the
//! encoder context to the clip length, which for 0.25s means
//! `audio_ctx = 13` -- a very small number to hand an encoder trained on
//! 1500-frame windows. This holds the audio fixed and sweeps only
//! `audio_ctx`, which is the only way to separate "short audio is slow"
//! from "small audio_ctx is slow."
//!
//! ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin cargo test --release -p asr --test small_audio_ctx_probe -- --ignored --nocapture

use std::time::Instant;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[test]
#[ignore]
fn sweep_audio_ctx_on_a_fixed_short_clip() {
    let Ok(model_path) = std::env::var("ASR_BENCH_MODEL_A") else {
        eprintln!("set ASR_BENCH_MODEL_A to run this probe");
        return;
    };

    // Fixed 0.25s of audio for every run -- only audio_ctx varies.
    let audio: Vec<f32> = (0..4_000).map(|i| (i as f32 * 0.05).sin() * 0.05).collect();

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("model should load");

    // 13 is what audio_ctx_for currently produces for this clip.
    for audio_ctx in [8, 12, 13, 14, 16, 18, 20, 22, 24, 32] {
        let mut state = ctx.create_state().expect("state");
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(8);
        params.set_no_context(true);
        params.set_no_timestamps(true);
        params.set_audio_ctx(audio_ctx);
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0); // isolate audio_ctx: no retry ladder
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        let start = Instant::now();
        state.full(params, &audio).expect("decode");
        let elapsed = start.elapsed();

        // Token count is the tell: if a degenerate encoder context makes
        // the decoder ramble until it hits its cap, that shows up here as
        // a large segment count, not just a big wall-clock number.
        let segments = state.full_n_segments();
        println!("audio_ctx {audio_ctx:>4}: {elapsed:>10.2?}  ({segments} segment(s))");
    }
}
