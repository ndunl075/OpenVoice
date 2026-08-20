//! The deadline race itself, kept independent of llama.cpp so it's
//! unit-testable with synthetic work instead of a real model.
//!
//! See `dictation-architecture.md` §2.4: "Hard deadline: 120ms. If it
//! returns in time, insert cleaned text. If not, insert raw and drop the
//! cleanup."
//!
//! "Drop" used to be literal: the timed-out generation was left running
//! to completion on its own thread with nobody reading the result. That
//! was measurably expensive rather than merely untidy --
//! `tests/abandoned_work_cost.rs` found a full generation takes
//! **516-677ms** against a **120ms** deadline, so *every* utterance left
//! ~400-550ms of LLM inference running purely to be thrown away. Latency
//! never showed it (the user already had their text); it showed up as
//! heat and fan noise.
//!
//! llama.cpp has no cancellation hook *inside* a single `decode` call,
//! but generation is a loop of many short decodes, so cooperative
//! cancellation between tokens gets almost all of it back. Hence
//! [`CancelToken`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

/// Cooperative cancellation signal handed to deadline-run work.
///
/// Checked *between* units of work (for the cleanup pass, between
/// generated tokens), so a cancelled job stops at the next boundary
/// rather than instantly -- which is enough, since the alternative was
/// running hundreds of milliseconds past the deadline.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// True once the deadline has passed and nobody will read the result.
    /// Work should return early -- whatever it returns is discarded.
    pub fn is_cancelled(&self) -> bool {
        // Relaxed is correct here: this is a one-way flag with no other
        // memory being published alongside it, and checking it in a hot
        // per-token loop shouldn't pay for a fence.
        self.0.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Runs `work` on a background thread and waits up to `deadline` for it to
/// finish. Returns `Some(result)` if it finished in time, `None` if the
/// deadline elapsed first -- in which case the work's [`CancelToken`] is
/// tripped so it can stop early instead of burning CPU on a result
/// nothing will read.
pub fn run_with_deadline<T: Send + 'static>(
    deadline: Duration,
    work: impl FnOnce(CancelToken) -> T + Send + 'static,
) -> Option<T> {
    let cancel = CancelToken::new();
    let worker_cancel = cancel.clone();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        // The receiver may already have given up and dropped `rx` by the
        // time this finishes; that's exactly the "drop the cleanup" case,
        // and a failed send here is the expected, silent outcome of it.
        let _ = tx.send(work(worker_cancel));
    });
    let result = rx.recv_timeout(deadline).ok();
    if result.is_none() {
        cancel.cancel();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn returns_the_result_when_work_finishes_in_time() {
        let result = run_with_deadline(Duration::from_millis(200), |_cancel| 42);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn returns_none_when_work_is_still_running_at_the_deadline() {
        let result = run_with_deadline(Duration::from_millis(30), |_cancel| {
            sleep(Duration::from_millis(300));
            "too slow"
        });
        assert_eq!(result, None);
    }

    #[test]
    fn work_is_not_cancelled_when_it_finishes_in_time() {
        let (probe_tx, probe_rx) = mpsc::channel();
        let result = run_with_deadline(Duration::from_millis(500), move |cancel| {
            let _ = probe_tx.send(cancel.is_cancelled());
            "in time"
        });
        assert_eq!(result, Some("in time"));
        assert_eq!(probe_rx.recv(), Ok(false), "should not be cancelled on the happy path");
    }

    #[test]
    fn timed_out_work_sees_its_cancel_token_trip() {
        // The actual point of CancelToken: without this, a timed-out
        // cleanup keeps generating tokens nobody will read (measured at
        // ~400-550ms of pure waste per utterance -- see this module's
        // docs).
        let (probe_tx, probe_rx) = mpsc::channel();
        let result = run_with_deadline(Duration::from_millis(20), move |cancel| {
            // Stand-in for the real per-token loop: poll until asked to stop.
            for _ in 0..200 {
                if cancel.is_cancelled() {
                    let _ = probe_tx.send("stopped early");
                    return "stopped early";
                }
                sleep(Duration::from_millis(5));
            }
            let _ = probe_tx.send("ran to completion");
            "ran to completion"
        });
        assert_eq!(result, None, "deadline should have elapsed first");
        assert_eq!(
            probe_rx.recv_timeout(Duration::from_millis(1000)),
            Ok("stopped early"),
            "cancelled work should bail out instead of running to completion"
        );
    }

    #[test]
    fn a_late_result_is_never_observed_after_giving_up() {
        // Regression guard for the "insert raw, then patch it later" bug
        // §2.4 explicitly warns against: once we've given up, the eventual
        // late result must not surface through some other channel.
        let (probe_tx, probe_rx) = mpsc::channel();
        let result = run_with_deadline(Duration::from_millis(20), move |_cancel| {
            sleep(Duration::from_millis(150));
            let _ = probe_tx.send("finished late");
            "finished late"
        });
        assert_eq!(result, None);
        // The background thread hasn't necessarily finished yet either --
        // confirm it eventually does, independently of our deadline logic.
        assert_eq!(
            probe_rx.recv_timeout(Duration::from_millis(500)),
            Ok("finished late")
        );
    }
}
