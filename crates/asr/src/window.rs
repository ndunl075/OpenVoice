//! Rolling-window scheduling, kept pure and independent of whisper-rs so
//! it's unit-testable without a model loaded.
//!
//! See `dictation-architecture.md` §2.3: "Rolling windows, decoded *while
//! the user speaks*. At release, only the final partial window is
//! outstanding." (The doc's illustrative numbers are 3s windows / 0.5s
//! overlap; [`WindowPolicy::default_16k`] tunes tighter than that once
//! decode speed allows -- see its doc comment.) This is the actual
//! latency win in the whole design (§1: "transcription should be nearly
//! finished before the user stops talking") -- everything else in this
//! crate just makes each individual decode fast; this is what makes most
//! of the decoding happen before the user is even done.

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

    /// 1.2s windows, 0.2s overlap (1.0s stride) at 16kHz -- tighter than
    /// the doc's illustrative "3s windows, 0.5s overlap" example, and the
    /// *overlap fraction* matters as much as the raw sizes here.
    ///
    /// Real measurement (`crates/asr/tests/real_time_factor.rs`) after the
    /// `audio_ctx` fix and the thread-count fix (see
    /// `asr::default_thread_count`'s doc comment) found `distil-small.en`
    /// decoding at roughly 0.8x real-time on this machine -- i.e. an
    /// N-second window takes about `0.8*N` seconds to decode. For
    /// streaming to actually keep up with a long, continuous utterance
    /// (not just look fine on short test clips), each window's decode
    /// needs to finish *before* the next window's worth of new audio has
    /// even arrived: `decode_time(window) < stride`, which with the 0.8x
    /// figure means `overlap < ~0.2 * window`. The doc's illustrative 3s
    /// window / 0.5s overlap (17%) is actually fine by that math, but an
    /// earlier retune to 1.5s/0.5s (33%) was not -- it would silently
    /// fall further behind over a long utterance, and since
    /// [`final_window`](Self::final_window) covers *everything* not yet
    /// windowed, that backlog would all land in the one decode call that
    /// blocks hotkey-release-to-insert latency, defeating the entire
    /// point of streaming ahead of time. 0.2/1.2 ≈ 17% leaves real
    /// margin while still landing a 1.0s final-tail bound -- tighter
    /// than the doc's 3s window would give.
    ///
    /// Caveat worth being honest about: the benchmark uses a synthetic
    /// swept sine, not real speech -- token count (and so decode time)
    /// can differ for real content. If it ever needs re-validating,
    /// that's what the benchmark test is for; don't just eyeball it.
    pub fn default_16k() -> Self {
        Self::new(16_000, Duration::from_millis(1200), Duration::from_millis(200))
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
    fn default_16k_keeps_overlap_fraction_low_enough_to_sustain_real_time() {
        let p = WindowPolicy::default_16k();
        assert_eq!(p.window_samples, 19_200); // 1.2s @ 16kHz
        assert_eq!(p.stride_samples, 16_000); // (1.2s - 0.2s) @ 16kHz -- the final-tail bound
        // The condition this whole tuning exists to satisfy: overlap
        // stays under ~20% of the window (see the doc comment above for
        // the 0.8x-real-time measurement this threshold comes from).
        let overlap_samples = p.window_samples - p.stride_samples;
        assert!(
            (overlap_samples as f64) < 0.2 * p.window_samples as f64,
            "overlap fraction too high to sustain real-time decode"
        );
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
