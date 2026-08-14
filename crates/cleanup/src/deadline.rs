//! The deadline race itself, kept independent of llama.cpp so it's
//! unit-testable with synthetic work instead of a real model.
//!
//! See `dictation-architecture.md` §2.4: "Hard deadline: 120ms. If it
//! returns in time, insert cleaned text. If not, insert raw and drop the
//! cleanup." The "drop" is literal here -- a timed-out generation isn't
//! cancelled (llama.cpp has no cooperative cancellation hook mid-decode),
//! it's abandoned on its own thread and its eventual result is never read.

use std::sync::mpsc;
use std::time::Duration;

/// Runs `work` on a background thread and waits up to `deadline` for it to
/// finish. Returns `Some(result)` if it finished in time, `None` if the
/// deadline elapsed first -- in which case the thread keeps running to
/// completion on its own, but nothing is waiting for it anymore.
pub fn run_with_deadline<T: Send + 'static>(
    deadline: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        // The receiver may already have given up and dropped `rx` by the
        // time this finishes; that's exactly the "drop the cleanup" case,
        // and a failed send here is the expected, silent outcome of it.
        let _ = tx.send(work());
    });
    rx.recv_timeout(deadline).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn returns_the_result_when_work_finishes_in_time() {
        let result = run_with_deadline(Duration::from_millis(200), || 42);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn returns_none_when_work_is_still_running_at_the_deadline() {
        let result = run_with_deadline(Duration::from_millis(30), || {
            sleep(Duration::from_millis(300));
            "too slow"
        });
        assert_eq!(result, None);
    }

    #[test]
    fn a_late_result_is_never_observed_after_giving_up() {
        // Regression guard for the "insert raw, then patch it later" bug
        // §2.4 explicitly warns against: once we've given up, the eventual
        // late result must not surface through some other channel.
        let (probe_tx, probe_rx) = mpsc::channel();
        let result = run_with_deadline(Duration::from_millis(20), move || {
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
