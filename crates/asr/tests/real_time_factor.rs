//! Permanent, `#[ignore]`d benchmark: how long does a *production*
//! `Transcriber::transcribe` call actually take for the window sizes
//! `WindowPolicy` schedules, on the real default model? Ad-hoc timing
//! claims aren't trustworthy (see `crates/asr/README.md`'s "Performance"
//! section on how the `audio_ctx` regression shipped for a while because
//! nobody had actually timed a real decode call) -- this is the tool for
//! answering "is a given window size fast enough" with a number instead
//! of a guess.
//!
//! Run with:
//! ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin cargo test --release -p asr --test real_time_factor -- --ignored --nocapture

use std::time::Instant;

use asr::{AsrConfig, Transcriber};

#[test]
#[ignore]
fn window_sizes_vs_wall_clock() {
    let Ok(model_path) = std::env::var("ASR_BENCH_MODEL_A") else {
        eprintln!("set ASR_BENCH_MODEL_A to run this benchmark");
        return;
    };

    // Real speech-ish signal (silence decodes unrealistically fast on some
    // models' VAD-adjacent heuristics) -- a simple swept sine is enough to
    // force real encoder+decoder work without needing an actual recording.
    let synth = |seconds: f64| -> Vec<f32> {
        let n = (seconds * 16_000.0) as usize;
        (0..n).map(|i| (i as f32 * 0.05).sin() * 0.05).collect()
    };

    // Thread count sweep: `AsrConfig::new`'s default clamps to `min(4,
    // available_parallelism)` (see `default_thread_count`'s doc comment),
    // matching whisper.cpp's own upstream default -- but that default
    // predates this machine's actual core count being checked. If more
    // threads meaningfully helps, that's a free win (no accuracy
    // tradeoff, unlike a smaller model); if it doesn't, the clamp is
    // already fine and the bottleneck is elsewhere.
    let available = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
    let mut thread_counts = vec![1, 4];
    if available > 4 {
        thread_counts.push(available / 2);
        thread_counts.push(available);
    }
    thread_counts.sort_unstable();
    thread_counts.dedup();

    for threads in thread_counts {
        let mut config = AsrConfig::new(&model_path);
        config.n_threads = threads;
        let transcriber = Transcriber::load(config).expect("model should load");

        for seconds in [1.5, 1.0, 0.7, 0.5] {
            let audio = synth(seconds);
            // Warm-up call: first decode after load pays one-time setup
            // cost that isn't representative of steady-state streaming.
            let _ = transcriber.transcribe(&audio);

            let start = Instant::now();
            let _ = transcriber.transcribe(&audio).expect("decode");
            let elapsed = start.elapsed();
            let rtf = elapsed.as_secs_f64() / seconds;
            println!(
                "threads={threads:<2} {seconds:>4}s window: {elapsed:?} wall-clock (real-time factor {rtf:.2}x, {} than real-time)",
                if rtf < 1.0 { "faster" } else { "slower" }
            );
        }
    }
}
