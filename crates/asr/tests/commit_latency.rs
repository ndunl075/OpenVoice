//! The number this whole project exists to hit: `dictation-architecture.md`'s
//! "< 200 ms from end-of-speech to text at cursor."
//!
//! Everything else in `crates/asr/tests/` measures throughput (can decode
//! keep up with speech?). This measures *latency at the moment the user
//! lets go of the key* -- a different question, and the one the README's
//! headline claim is about. At hotkey release, the outstanding work is:
//!
//!   1. decode the trailing partial window (`WindowPolicy::final_window`),
//!      which covers everything since the last full window's stride --
//!      so worst case is one full stride of audio, average ~half that
//!   2. optionally, the cleanup pass (hard-deadlined at 120ms, §2.4)
//!   3. clipboard-swap injection (~15ms per the doc's table)
//!
//! Step 1 is the variable one and the only one this test measures. Run:
//!
//! ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin cargo test --release -p asr --test commit_latency -- --ignored --nocapture

use std::time::{Duration, Instant};

use asr::{AsrConfig, Transcriber, WindowPolicy};

/// §2.4's cleanup deadline, duplicated here rather than depending on the
/// `cleanup` crate (which would drag llama.cpp into this test's build for
/// one constant).
const CLEANUP_DEADLINE_MS: u128 = 120;
/// The doc's own estimate for clipboard-swap injection (§1's table).
const INJECT_MS: u128 = 15;
/// `dictation-architecture.md`'s headline target.
const TARGET_MS: u128 = 200;

#[test]
#[ignore]
fn tail_decode_fits_the_200ms_end_to_end_budget() {
    let Ok(model_path) = std::env::var("ASR_BENCH_MODEL_A") else {
        eprintln!("set ASR_BENCH_MODEL_A to run this benchmark");
        return;
    };

    let policy = WindowPolicy::default_16k();
    let stride_secs = policy.stride_samples as f64 / 16_000.0;

    let synth = |seconds: f64| -> Vec<f32> {
        let n = (seconds * 16_000.0) as usize;
        (0..n).map(|i| (i as f32 * 0.05).sin() * 0.05).collect()
    };

    println!("WindowPolicy stride = {stride_secs:.2}s -- the worst-case trailing tail at release.\n");

    // The tail is whatever arrived since the last full window's stride:
    // uniformly distributed in [0, stride], so measure across that range
    // rather than cherry-picking the flattering end.
    let tails: Vec<f64> = vec![0.1, 0.25, stride_secs / 2.0, stride_secs];

    // Sweeping temperature_inc, not just reporting one config: 0.0 is
    // the architecture doc's "no temperature fallback" (fastest, but
    // shipped real garbage output -- see AsrConfig::temperature_inc),
    // 0.2 is whisper.cpp's own default, 0.4 is this crate's compromise.
    // The point is to show what the retry ladder actually costs rather
    // than asserting it from the source.
    for temperature_inc in [0.0_f32, 0.2, 0.4] {
        let mut config = AsrConfig::new(&model_path);
        config.temperature_inc = temperature_inc;
        let transcriber = Transcriber::load(config).expect("model should load");

        println!("--- temperature_inc = {temperature_inc} ---");
        let mut worst_total = 0u128;
        for &tail_secs in &tails {
            let audio = synth(tail_secs);
            let _ = transcriber.transcribe(&audio); // warm-up

            // Best of a few runs: this is a latency *floor* question, and
            // scheduler noise only ever adds. Being generous here is the
            // honest direction -- if even the best case misses the
            // target, the claim is dead regardless of noise.
            let mut best = Duration::MAX;
            for _ in 0..3 {
                let start = Instant::now();
                let _ = transcriber.transcribe(&audio).expect("decode");
                best = best.min(start.elapsed());
            }

            let decode_ms = best.as_millis();
            let with_inject = decode_ms + INJECT_MS;
            let with_cleanup = with_inject + CLEANUP_DEADLINE_MS;
            worst_total = worst_total.max(with_cleanup);

            println!(
                "  tail {tail_secs:>4.2}s: decode {decode_ms:>5}ms | +inject {with_inject:>5}ms | +cleanup(worst) {with_cleanup:>5}ms  {}",
                if with_cleanup <= TARGET_MS {
                    "OK"
                } else if with_inject <= TARGET_MS {
                    "OVER (only without cleanup)"
                } else {
                    "OVER"
                }
            );
        }
        println!("  worst end-of-speech -> cursor: ~{worst_total}ms vs {TARGET_MS}ms target\n");
    }

    println!(
        "This test deliberately does not assert -- it reports. See crates/asr/README.md \
         and the root README's latency section for what the real numbers mean."
    );
}
