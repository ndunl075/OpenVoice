//! Rolling-window scheduling, kept pure and independent of whisper-rs so
//! it's unit-testable without a model loaded.
//!
//! See `dictation-architecture.md` §2.3: "Rolling 3s windows with 0.5s
//! overlap, decoded *while the user speaks*. At release, only the final
//! partial window is outstanding." This is the actual latency win in the
//! whole design (§1: "transcription should be nearly finished before the
//! user stops talking") -- everything else in this crate just makes each
//! individual decode fast; this is what makes most of the decoding happen
//! before the user is even done.

use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct WindowPolicy {
    pub window_samples: usize,
    /// `window_samples - overlap_samples`: how far the start of each
    /// window advances from the previous one.
    pub stride_samples: usize,
}

impl WindowPolicy {
    pub fn new(sample_rate_hz: u32, window: Duration, overlap: Duration) -> Self {
        let window_samples = (sample_rate_hz as f64 * window.as_secs_f64()).round() as usize;
        let overlap_samples = (sample_rate_hz as f64 * overlap.as_secs_f64()).round() as usize;
        Self {
            window_samples,
            stride_samples: window_samples.saturating_sub(overlap_samples).max(1),
        }
    }

    /// 1.5s windows, 0.5s overlap, at 16kHz -- tighter than the doc's
    /// illustrative "3s windows" example. Once `asr::audio_ctx` fixed the
    /// real bottleneck (see that module's docs: whisper.cpp was paying for
    /// a full 30s encoder context on every decode regardless of input
    /// length), decode of `distil-small.en` sits close to real-time, which
    /// makes a shorter stride a pure win rather than a tradeoff: the first
    /// partial result lands sooner, the pill's incremental updates feel
    /// more alive, and -- the part that actually matters for perceived
    /// latency -- the trailing partial window left to decode at hotkey
    /// release ([`final_window`](Self::final_window)) is bounded by one
    /// stride (1s) instead of three.
    pub fn default_16k() -> Self {
        Self::new(16_000, Duration::from_millis(1500), Duration::from_millis(500))
    }

    /// The `[start, end)` sample range of the next full window, if enough
    /// audio has accumulated to decode one yet. `windows_decoded` is how
    /// many full windows this session has already consumed.
    pub fn next_window(&self, captured_samples: usize, windows_decoded: usize) -> Option<(usize, usize)> {
        let start = windows_decoded * self.stride_samples;
        let end = start + self.window_samples;
        (end <= captured_samples).then_some((start, end))
    }

    /// The trailing partial window at end-of-speech: from where the last
    /// full window's stride left off, to whatever's been captured since.
    /// This is the "only the final partial window is outstanding" piece --
    /// everything before `start` was already decoded while the user was
    /// still talking.
    pub fn final_window(&self, captured_samples: usize, windows_decoded: usize) -> Option<(usize, usize)> {
        let start = windows_decoded * self.stride_samples;
        (start < captured_samples).then_some((start, captured_samples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> WindowPolicy {
        // Small numbers for readable tests: 30-sample windows, 10-sample overlap.
        WindowPolicy {
            window_samples: 30,
            stride_samples: 20,
        }
    }

    #[test]
    fn default_16k_is_a_tight_window_bounded_final_tail() {
        let p = WindowPolicy::default_16k();
        assert_eq!(p.window_samples, 24_000); // 1.5s @ 16kHz
        assert_eq!(p.stride_samples, 16_000); // (1.5s - 0.5s) @ 16kHz -- the final-window bound
    }

    #[test]
    fn no_window_ready_until_enough_audio_captured() {
        let p = policy();
        assert_eq!(p.next_window(29, 0), None);
        assert_eq!(p.next_window(30, 0), Some((0, 30)));
    }

    #[test]
    fn second_window_starts_at_the_stride_not_the_window_end() {
        let p = policy();
        // First window [0,30) already decoded; second starts at stride=20.
        assert_eq!(p.next_window(49, 1), None);
        assert_eq!(p.next_window(50, 1), Some((20, 50)));
    }

    #[test]
    fn windows_overlap_by_window_minus_stride() {
        let p = policy();
        let (_s0, e0) = p.next_window(30, 0).unwrap();
        let (s1, _e1) = p.next_window(50, 1).unwrap();
        assert_eq!(e0 - s1, 10); // 30 - 20 = 10-sample overlap, as configured
    }

    #[test]
    fn final_window_covers_everything_since_the_last_stride() {
        let p = policy();
        // One full window decoded (windows_decoded=1, i.e. stride start = 20);
        // 15 more samples captured since (total 65) but not enough for window 2.
        assert_eq!(p.final_window(65, 1), Some((20, 65)));
    }

    #[test]
    fn final_window_is_none_when_nothing_new_since_last_window() {
        let p = policy();
        assert_eq!(p.final_window(20, 1), None);
    }

    #[test]
    fn zero_overlap_never_divides_stride_to_zero() {
        // window == overlap would otherwise make stride 0 and loop forever.
        let p = WindowPolicy::new(16_000, Duration::from_millis(500), Duration::from_millis(500));
        assert!(p.stride_samples >= 1);
    }
}
