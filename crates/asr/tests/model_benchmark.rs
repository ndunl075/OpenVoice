//! Ad-hoc wall-clock benchmark comparing two ASR models' decode time on the
//! same synthetic audio. Not run in CI (needs local model files) -- run by
//! hand with:
//!
//! ```sh
//! ASR_BENCH_MODEL_A=models/ggml-small.en-q5_1.bin \
//! ASR_BENCH_MODEL_B=models/ggml-distil-small.en.bin \
//! cargo test --release -p asr --test model_benchmark -- --ignored --nocapture
//! ```

use std::time::Instant;

#[test]
#[ignore]
fn compare_two_models() {
    let (Ok(a_path), Ok(b_path)) = (
        std::env::var("ASR_BENCH_MODEL_A"),
        std::env::var("ASR_BENCH_MODEL_B"),
    ) else {
        eprintln!("set ASR_BENCH_MODEL_A and ASR_BENCH_MODEL_B to run this benchmark");
        return;
    };

    // Doesn't need to be real speech -- we're timing decode wall-clock,
    // not measuring accuracy. A few seconds of low-amplitude tone is a
    // stand-in for "some audio arrived."
    let three_seconds: Vec<f32> = (0..3 * 16_000)
        .map(|i| (i as f32 * 0.05).sin() * 0.05)
        .collect();

    for (label, path) in [("A", a_path), ("B", b_path)] {
        let load_start = Instant::now();
        let transcriber = asr::Transcriber::load(asr::AsrConfig::new(&path))
            .unwrap_or_else(|e| panic!("failed to load {label} ({path}): {e}"));
        let load_time = load_start.elapsed();

        // Warm-up run: first decode pays one-time setup cost (mel filter
        // bank init, thread pool spin-up) we don't want polluting the
        // measurement below.
        let _ = transcriber.transcribe(&three_seconds);

        let decode_start = Instant::now();
        let _ = transcriber.transcribe(&three_seconds);
        let decode_time = decode_start.elapsed();

        println!("{label} ({path}): load={load_time:?} decode_3s={decode_time:?}");
    }
}
