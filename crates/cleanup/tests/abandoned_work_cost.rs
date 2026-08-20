//! How much CPU does a *timed-out* cleanup actually burn?
//!
//! §2.4's deadline is 120ms, and `run_with_deadline` abandons anything
//! slower -- but "abandon" only means "stop waiting for it". The
//! generation thread keeps decoding tokens to completion with nobody
//! reading the result. If a full generation takes far longer than the
//! deadline, every timed-out utterance silently burns the difference,
//! which shows up as fan noise rather than as latency (the user already
//! got their text).
//!
//! This measures the gap. Run:
//!
//! CLEANUP_BENCH_MODEL=models/qwen2.5-0.5b-instruct-q4_k_m.gguf cargo test --release -p cleanup --test abandoned_work_cost -- --ignored --nocapture

use std::time::{Duration, Instant};

use cleanup::{CleanupModel, CLEANUP_DEADLINE};

#[test]
#[ignore]
fn full_generation_time_versus_the_deadline() {
    let Ok(model_path) = std::env::var("CLEANUP_BENCH_MODEL") else {
        eprintln!("set CLEANUP_BENCH_MODEL to run this benchmark");
        return;
    };

    let model = match CleanupModel::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("couldn't load cleanup model: {e}");
            return;
        }
    };

    // Representative raw ASR output: disfluent, unpunctuated -- exactly
    // what this pass exists to fix.
    let samples = [
        "um so i think we should probably ship it on friday",
        "yeah uh lets go with the second option i mean the one with the cache",
        "can you send me that link about the the deployment thing",
    ];

    for raw in samples {
        // A deliberately huge deadline, so this measures how long the
        // generation *actually* takes rather than how long we wait.
        let start = Instant::now();
        let full = model.clean_with_deadline(raw, Duration::from_secs(60));
        let full_ms = start.elapsed().as_millis();

        let deadline_ms = CLEANUP_DEADLINE.as_millis();
        let wasted = full_ms.saturating_sub(deadline_ms);
        println!(
            "full generation {full_ms:>5}ms | deadline {deadline_ms}ms | wasted-if-timed-out {wasted:>5}ms  -> {:?}",
            full.as_deref().unwrap_or("(failed)")
        );
    }

    println!(
        "\n'wasted-if-timed-out' is CPU burned after the user already has their text: \
         the result is never read. Anything much above zero is pure heat."
    );
}
