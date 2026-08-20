//! Scopes whisper.cpp's encoder context to how much audio actually arrived,
//! instead of the library default.
//!
//! This is the single biggest speed lever in this crate, bigger than any
//! decode-parameter tuning in `lib.rs`: whisper's encoder architecture is
//! trained on fixed 30-second windows, and whisper.cpp's `audio_ctx`
//! parameter defaults to `0`, which means "use the full 30-second
//! context" -- **regardless of how much actual audio you pass in**. Feed
//! it 3 seconds of audio with the default and it still runs the encoder
//! over a context sized for 30 seconds, wasting roughly 10x the necessary
//! compute. Measured on real hardware: a 3-second clip that took ~62s to
//! decode at the default dropped to ~5s once `audio_ctx` was scoped to the
//! actual clip length -- an ~11.7x speedup for zero accuracy cost, since
//! it's not skipping anything, just not wastefully padding.
//!
//! whisper.cpp's encoder produces 50 output frames per second of audio
//! (its mel/conv frontend downsamples 100 mel-frames/sec by 2x), so
//! `audio_ctx` is specified in units of 1/50th of a second, capped at
//! 1500 (= 30s, the model's trained maximum).

const FRAMES_PER_SECOND: f64 = 50.0;
const MAX_CONTEXT_FRAMES: i32 = 1500; // 30s -- whisper's trained maximum

/// Round `audio_ctx` up to a multiple of this, and never go below it.
///
/// Not a guess -- measured. `crates/asr/tests/small_audio_ctx_probe.rs`
/// sweeps `audio_ctx` over a *fixed* 0.25s clip (so only the parameter
/// varies) and finds a stark, non-monotonic split:
///
/// ```text
/// audio_ctx  8: 3.85s     audio_ctx 16: 327ms
/// audio_ctx 12: 4.93s     audio_ctx 24: 432ms
/// audio_ctx 13: 5.21s     audio_ctx 32: 578ms
/// audio_ctx 14: 5.06s
/// audio_ctx 18: 5.27s
/// audio_ctx 20: 4.37s
/// audio_ctx 22: 4.44s
/// ```
///
/// The fast values are exactly the multiples of 8 at or above 16; every
/// other small value costs **roughly 10x more**, on identical audio.
/// That signature -- a hard fast/slow split on block-size boundaries
/// rather than a smooth curve -- is ggml falling off its SIMD-blocked
/// matmul kernels onto a scalar path, consistent with the AVX2/FMA
/// flags this build already has to force on explicitly (see the repo's
/// `.cargo/config.toml`).
///
/// This mattered in the one place latency is actually felt: the
/// trailing partial window at hotkey release (`WindowPolicy::final_window`)
/// is short by construction, so it landed in the pathological range
/// constantly.
const CONTEXT_BLOCK: i32 = 16;

/// The `audio_ctx` value to pass to whisper.cpp for `samples` of mono
/// audio at `sample_rate_hz`: enough frames to cover the audio, rounded
/// up to a [`CONTEXT_BLOCK`] boundary (see that constant for the
/// measurements behind it), and clamped to whisper's 30s maximum.
///
/// Rounding *up* is always safe -- a context larger than the audio just
/// means the encoder sees padding, exactly as it would for any clip
/// shorter than its window. Rounding down would truncate real audio.
pub fn audio_ctx_for(samples: usize, sample_rate_hz: u32) -> i32 {
    let seconds = samples as f64 / sample_rate_hz as f64;
    let frames = (seconds * FRAMES_PER_SECOND).ceil() as i32;
    // Round up to the next CONTEXT_BLOCK boundary. Written out rather
    // than via `div_ceil`, which isn't stable on this toolchain.
    let blocked = ((frames + CONTEXT_BLOCK - 1) / CONTEXT_BLOCK) * CONTEXT_BLOCK;
    blocked.clamp(CONTEXT_BLOCK, MAX_CONTEXT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_seconds_at_16k_rounds_150_frames_up_to_a_block() {
        // 3s = 150 frames -> next multiple of 16 is 160.
        assert_eq!(audio_ctx_for(3 * 16_000, 16_000), 160);
    }

    #[test]
    fn one_second_rounds_50_frames_up_to_a_block() {
        assert_eq!(audio_ctx_for(16_000, 16_000), 64); // 50 -> 64
    }

    #[test]
    fn never_returns_a_context_smaller_than_the_audio() {
        // The invariant that keeps rounding-to-a-block from silently
        // truncating: whatever we return must still cover the real
        // frame count.
        for samples in [1, 100, 4_000, 16_001, 24_000, 48_000, 10 * 16_000] {
            let exact = ((samples as f64 / 16_000.0) * FRAMES_PER_SECOND).ceil() as i32;
            assert!(
                audio_ctx_for(samples, 16_000) >= exact.min(MAX_CONTEXT_FRAMES),
                "context smaller than the audio for {samples} samples"
            );
        }
    }

    #[test]
    fn every_result_is_block_aligned_and_never_in_the_slow_range() {
        // The whole point of CONTEXT_BLOCK -- see its doc comment for the
        // ~10x measurements. Values below 16, or not a multiple of 16,
        // are the pathological ones.
        for samples in [0, 1, 800, 4_000, 8_000, 16_000, 24_000, 100_000] {
            let ctx = audio_ctx_for(samples, 16_000);
            assert!(ctx >= CONTEXT_BLOCK, "{ctx} is below the safe floor");
            assert_eq!(ctx % CONTEXT_BLOCK, 0, "{ctx} is not block-aligned");
        }
    }

    #[test]
    fn empty_audio_still_gets_a_positive_block_aligned_context() {
        assert_eq!(audio_ctx_for(0, 16_000), CONTEXT_BLOCK);
    }

    #[test]
    fn thirty_seconds_or_more_clamps_to_the_trained_maximum() {
        // 1500 is itself not a multiple of 16, but it's whisper's hard
        // architectural maximum -- the clamp has to win over alignment.
        assert_eq!(audio_ctx_for(30 * 16_000, 16_000), MAX_CONTEXT_FRAMES);
        assert_eq!(audio_ctx_for(60 * 16_000, 16_000), MAX_CONTEXT_FRAMES);
    }

    #[test]
    fn scales_correctly_at_a_different_sample_rate() {
        // Same duration, different rate -- frame count should match by
        // time, not raw sample count.
        assert_eq!(audio_ctx_for(3 * 48_000, 48_000), 160);
    }
}
