//! Always-on, in-memory audio ring buffer.
//!
//! See `dictation-architecture.md` §2.1. The daemon keeps this buffer full
//! of the last ~30s of microphone audio at all times so a hotkey press never
//! pays mic-device-init latency, and so pre-roll (capturing the moment
//! *before* the hotkey) is possible at all.
//!
//! Deliberately dependency-free: no file I/O, no network, nothing that could
//! let audio escape the process. That's a design constraint worth enforcing
//! in code, not just documentation — see the crate's `Cargo.toml`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Default capacity per §2.1: "30s circ." at 16kHz mono.
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_CAPACITY: Duration = Duration::from_secs(30);

/// An opaque point in the buffer's timeline, produced by [`RingBuffer::mark`]
/// and consumed by [`RingBuffer::read_since`]. Lets a caller say "give me
/// everything captured since I pressed the hotkey" without copying audio on
/// every callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mark(u64);

/// Fixed-capacity circular buffer of mono `f32` PCM samples.
///
/// Once full, pushing overwrites the oldest samples in place — there is no
/// growth, no heap churn per callback, and nothing is ever written to disk.
pub struct RingBuffer {
    data: Vec<f32>,
    capacity: usize,
    /// Monotonic count of every sample ever pushed. Used instead of a plain
    /// write cursor so `Mark`s stay unambiguous across wraparound.
    total_written: u64,
}

impl RingBuffer {
    /// Creates a buffer that holds exactly `capacity_samples` samples.
    ///
    /// # Panics
    /// Panics if `capacity_samples` is zero — a zero-capacity ring buffer
    /// can't hold a mark's worth of anything and almost certainly indicates
    /// a misconfiguration.
    pub fn new(capacity_samples: usize) -> Self {
        assert!(capacity_samples > 0, "ring buffer capacity must be > 0");
        Self {
            data: vec![0.0; capacity_samples],
            capacity: capacity_samples,
            total_written: 0,
        }
    }

    /// Creates a buffer sized for `duration` of audio at `sample_rate_hz`.
    pub fn with_duration(sample_rate_hz: u32, duration: Duration) -> Self {
        let samples = (sample_rate_hz as f64 * duration.as_secs_f64()).round() as usize;
        Self::new(samples.max(1))
    }

    /// Convenience: the default 30s @ 16kHz buffer described in §2.1.
    pub fn default_mono_16k() -> Self {
        Self::with_duration(DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CAPACITY)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How many samples have ever been written (not clamped to capacity) —
    /// useful for tests and diagnostics.
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// How many *currently retained* samples are available to read.
    pub fn len(&self) -> usize {
        (self.total_written.min(self.capacity as u64)) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.total_written == 0
    }

    /// Appends samples from a mic callback, overwriting the oldest audio
    /// once the buffer is full. This is the only mutation the audio
    /// callback thread needs to perform.
    pub fn push(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        // If the incoming chunk alone is >= capacity, only its tail matters.
        let samples = if samples.len() > self.capacity {
            &samples[samples.len() - self.capacity..]
        } else {
            samples
        };

        let start = (self.total_written as usize) % self.capacity;
        let first_len = samples.len().min(self.capacity - start);
        self.data[start..start + first_len].copy_from_slice(&samples[..first_len]);
        if first_len < samples.len() {
            let rest = &samples[first_len..];
            self.data[..rest.len()].copy_from_slice(rest);
        }
        self.total_written += samples.len() as u64;
    }

    /// A marker for "now" in the buffer's timeline. Pair with
    /// [`read_since`](Self::read_since) to pull exactly what was captured
    /// during a recording session (e.g. hotkey-down to hotkey-up).
    pub fn mark(&self) -> Mark {
        Mark(self.total_written)
    }

    /// Samples pushed since `mark`, oldest first.
    ///
    /// If more than [`capacity`](Self::capacity) samples were pushed since
    /// the mark, the oldest of them have already been overwritten; this
    /// returns only what's still retained (i.e. it saturates rather than
    /// panicking or fabricating data).
    pub fn read_since(&self, mark: Mark) -> Vec<f32> {
        let since = self.total_written.saturating_sub(mark.0);
        let n = since.min(self.capacity as u64) as usize;
        self.read_last(n)
    }

    /// The most recent `n_samples` of retained audio, oldest first. Used
    /// directly for pre-roll (§2.1: "capture the ~500ms before the hotkey
    /// press").
    pub fn read_last(&self, n_samples: usize) -> Vec<f32> {
        let n = n_samples.min(self.len());
        let start_total = self.total_written - n as u64;
        let mut out = Vec::with_capacity(n);
        let start_idx = (start_total as usize) % self.capacity;
        let first_len = n.min(self.capacity - start_idx);
        out.extend_from_slice(&self.data[start_idx..start_idx + first_len]);
        if first_len < n {
            out.extend_from_slice(&self.data[..n - first_len]);
        }
        out
    }

    /// The most recent `duration` of retained audio at `sample_rate_hz`.
    pub fn read_last_duration(&self, sample_rate_hz: u32, duration: Duration) -> Vec<f32> {
        let n = (sample_rate_hz as f64 * duration.as_secs_f64()).round() as usize;
        self.read_last(n)
    }
}

/// A `RingBuffer` shared between the audio callback thread (writer) and the
/// hotkey/VAD/ASR threads (readers). `audio-input` owns construction of the
/// writer side; everything downstream just holds one of these.
pub type SharedRingBuffer = Arc<Mutex<RingBuffer>>;

pub fn shared_default_mono_16k() -> SharedRingBuffer {
    Arc::new(Mutex::new(RingBuffer::default_mono_16k()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let rb = RingBuffer::new(10);
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.read_last(5), Vec::<f32>::new());
    }

    #[test]
    fn read_last_returns_what_was_pushed_when_under_capacity() {
        let mut rb = RingBuffer::new(10);
        rb.push(&[1.0, 2.0, 3.0]);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.read_last(3), vec![1.0, 2.0, 3.0]);
        assert_eq!(rb.read_last(10), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn overwrites_oldest_samples_once_full() {
        let mut rb = RingBuffer::new(4);
        rb.push(&[1.0, 2.0, 3.0, 4.0]);
        rb.push(&[5.0, 6.0]);
        // capacity 4, total written 6 -> retains samples 3,4,5,6
        assert_eq!(rb.len(), 4);
        assert_eq!(rb.read_last(4), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn push_larger_than_capacity_keeps_only_the_tail() {
        let mut rb = RingBuffer::new(3);
        rb.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(rb.read_last(3), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn many_small_pushes_match_one_big_push() {
        let mut a = RingBuffer::new(8);
        let mut b = RingBuffer::new(8);
        let data: Vec<f32> = (1..=20).map(|x| x as f32).collect();
        a.push(&data);
        for chunk in data.chunks(3) {
            b.push(chunk);
        }
        assert_eq!(a.read_last(8), b.read_last(8));
    }

    #[test]
    fn mark_and_read_since_captures_exactly_the_session() {
        let mut rb = RingBuffer::new(100);
        rb.push(&[0.0; 10]); // unrelated audio before the "hotkey press"
        let mark = rb.mark();
        rb.push(&[1.0, 2.0, 3.0]);
        rb.push(&[4.0, 5.0]);
        assert_eq!(rb.read_since(mark), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn read_since_saturates_when_session_overruns_capacity() {
        let mut rb = RingBuffer::new(4);
        let mark = rb.mark();
        rb.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 6 samples into a 4-cap buffer
        // Oldest 2 of the session are already gone; only the last 4 remain.
        assert_eq!(rb.read_since(mark), vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn read_since_immediately_after_mark_is_empty() {
        let mut rb = RingBuffer::new(10);
        rb.push(&[1.0, 2.0]);
        let mark = rb.mark();
        assert_eq!(rb.read_since(mark), Vec::<f32>::new());
    }

    #[test]
    fn with_duration_computes_default_16k_30s_size() {
        let rb = RingBuffer::with_duration(DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CAPACITY);
        assert_eq!(rb.capacity(), 480_000);
        let default_rb = RingBuffer::default_mono_16k();
        assert_eq!(default_rb.capacity(), 480_000);
    }

    #[test]
    fn read_last_duration_converts_seconds_to_samples() {
        let mut rb = RingBuffer::with_duration(16_000, Duration::from_secs(2));
        let data: Vec<f32> = (0..32_000).map(|i| i as f32).collect();
        rb.push(&data);
        let last_500ms = rb.read_last_duration(16_000, Duration::from_millis(500));
        assert_eq!(last_500ms.len(), 8_000);
        assert_eq!(last_500ms.first(), Some(&24_000.0));
        assert_eq!(last_500ms.last(), Some(&31_999.0));
    }

    #[test]
    fn shared_default_is_usable_across_a_mutex() {
        let shared = shared_default_mono_16k();
        {
            let mut guard = shared.lock().unwrap();
            guard.push(&[1.0, 2.0, 3.0]);
        }
        let guard = shared.lock().unwrap();
        assert_eq!(guard.read_last(3), vec![1.0, 2.0, 3.0]);
    }
}
