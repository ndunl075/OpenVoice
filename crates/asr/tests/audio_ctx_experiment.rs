//! Experimental: does setting `audio_ctx` proportional to actual audio
//! length avoid whisper.cpp encoding a full 30s-equivalent context for
//! short clips? Run with:
//!
//! ASR_BENCH_MODEL_A=<path> cargo test --release -p asr --test audio_ctx_experiment -- --ignored --nocapture

use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[test]
#[ignore]
fn compare_default_vs_scoped_audio_ctx() {
    let Ok(model_path) = std::env::var("ASR_BENCH_MODEL_A") else {
        eprintln!("set ASR_BENCH_MODEL_A");
        return;
    };

    let three_seconds: Vec<f32> = (0..3 * 16_000).map(|i| (i as f32 * 0.05).sin() * 0.05).collect();

    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .expect("model should load");

    for (label, audio_ctx) in [("default (0 = full 30s)", 0), ("scoped to ~3s (150)", 150)] {
        let mut state = ctx.create_state().expect("state");
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(4);
        params.set_no_context(true);
        params.set_no_timestamps(true);
        params.set_audio_ctx(audio_ctx);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);

        let start = Instant::now();
        state.full(params, &three_seconds).expect("decode");
        println!("{label}: {:?}", start.elapsed());
    }
}
