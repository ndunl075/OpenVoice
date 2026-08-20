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

/// **The floor that keeps output correct.** Never emit an `audio_ctx`
/// below this, however short the clip.
///
/// This is the correction to a genuinely bad bug. `audio_ctx` scoping was
/// originally introduced and tuned purely on wall-clock time, measured
/// against a *synthetic sine wave* -- and nobody ever looked at the text
/// coming out. Scoped tightly to clip length, it shipped repetition
/// loops in production: a two-second "hey what's going on" came out as
/// that phrase repeated ~30 times, degrading into "What? What? What?".
///
/// Measured properly this time -- real speech, known ground truth,
/// asserting on the transcript rather than the clock
/// (`crates/asr/tests/audio_ctx_quality.rs`), on a 2.51s utterance:
///
/// ```text
/// audio_ctx   64: 4.25s   unique-word ratio 0.03  REPETITION LOOP
/// audio_ctx  128: 2.15s   unique-word ratio 0.31  REPETITION LOOP
/// audio_ctx  256: 3.84s   unique-word ratio 1.00  correct
/// audio_ctx 1500: 29.8s   unique-word ratio 1.00  correct  (whisper's default)
/// ```
///
/// Verified on a 0.6s clip too, since the trailing window at hotkey
/// release is short by construction and short clips degenerate worst:
/// 64 still loops, 256 is clean.
///
/// Whisper's encoder is trained on fixed 1500-frame windows; shrinking
/// its context far below that takes the model out of distribution, and
/// the decoder starts looping. 256 is the smallest value measured clean
/// at both lengths, so it's the floor -- with the honest caveat that
/// it's an empirical threshold from two fixtures, not a value derived
/// from the architecture. Treat it as "known safe here," not "provably
/// minimal."
///
/// `dictation-architecture.md` §5 called this exact failure: "If you win
/// on milliseconds and lose on output quality, you have built a worse
/// product with a better benchmark."
const MIN_CONTEXT_FRAMES: i32 = 256;

/// Round `audio_ctx` up to a multiple of this.
///
/// Separate, smaller effect from [`MIN_CONTEXT_FRAMES`], measured in
/// `crates/asr/tests/small_audio_ctx_probe.rs`: on identical audio,
/// values that weren't multiples of 16 could cost ~10x more time than
/// neighbouring aligned ones -- ggml falling off its SIMD-blocked matmul
/// kernels onto a scalar path. Those measurements were taken at values
/// (8-32) that the floor above now excludes entirely, so this mostly
/// just keeps the encoder on aligned sizes rather than fixing anything
/// on its own. Harmless to keep, and cheap insurance if the floor is
/// ever revisited.
const CONTEXT_BLOCK: i32 = 16;

/// The `audio_ctx` value to pass to whisper.cpp for `samples` of mono
/// audio at `sample_rate_hz`: enough frames to cover the audio, rounded
/// up to a [`CONTEXT_BLOCK`] boundary, and clamped to
/// `[MIN_CONTEXT_FRAMES, MAX_CONTEXT_FRAMES]`.
///
/// Rounding *up* is always safe -- a context larger than the audio just
/// means the encoder sees padding, exactly as it would for any clip
/// shorter than its window. Rounding *down* is what breaks things: see
/// [`MIN_CONTEXT_FRAMES`].
pub fn audio_ctx_for(samples: usize, sample_rate_hz: u32) -> i32 {
    let seconds = samples as f64 / sample_rate_hz as f64;
    let frames = (seconds * FRAMES_PER_SECOND).ceil() as i32;
    // Round up to the next CONTEXT_BLOCK boundary. Written out rather
    // than via `div_ceil`, which isn't stable on this toolchain.
    let blocked = ((frames + CONTEXT_BLOCK - 1) / CONTEXT_BLOCK) * CONTEXT_BLOCK;
    blocked.clamp(MIN_CONTEXT_FRAMES, MAX_CONTEXT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_clips_are_lifted_to_the_safe_floor_not_scoped_tight() {
        // The regression this whole floor exists for: 3s of audio only
        // *needs* 150 frames, and 1s only needs 50, but both of those
        // produce repetition loops. See MIN_CONTEXT_FRAMES.
        assert_eq!(audio_ctx_for(3 * 16_000, 16_000), MIN_CONTEXT_FRAMES);
        assert_eq!(audio_ctx_for(16_000, 16_000), MIN_CONTEXT_FRAMES);
        assert_eq!(audio_ctx_for(4_000, 16_000), MIN_CONTEXT_FRAMES);
    }

    #[test]
    fn longer_clips_still_scope_above_the_floor() {
        // Past the floor, scoping still does its job -- a 10s clip
        // shouldn't pay for whisper's full 30s context.
        let ctx = audio_ctx_for(10 * 16_000, 16_000);
        assert!(ctx > MIN_CONTEXT_FRAMES, "should scope above the floor, got {ctx}");
        assert!(ctx < MAX_CONTEXT_FRAMES, "should still be cheaper than the 30s default");
    }

    #[test]
    fn never_returns_a_value_in_the_repetition_loop_range() {
        // Guard against anyone re-tightening this for a speed win
        // without re-running the *quality* benchmark.
        for samples in [0, 1, 4_000, 9_600, 16_000, 40_160, 5 * 16_000] {
            assert!(
                audio_ctx_for(samples, 16_000) >= MIN_CONTEXT_FRAMES,
                "below the measured-safe floor for {samples} samples"
            );
        }
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
    fn every_result_is_block_aligned() {
        for samples in [0, 1, 800, 4_000, 8_000, 16_000, 24_000, 100_000] {
            let ctx = audio_ctx_for(samples, 16_000);
            assert_eq!(ctx % CONTEXT_BLOCK, 0, "{ctx} is not block-aligned");
        }
    }

    #[test]
    fn empty_audio_still_gets_the_safe_floor() {
        assert_eq!(audio_ctx_for(0, 16_000), MIN_CONTEXT_FRAMES);
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
        // Same duration, different rate -> same answer: frame count is
        // derived from time, not raw sample count. (Both land on the
        // floor at this duration; the 10s case below is what actually
        // exercises the scaling above it.)
        assert_eq!(
            audio_ctx_for(3 * 48_000, 48_000),
            audio_ctx_for(3 * 16_000, 16_000)
        );
        assert_eq!(
            audio_ctx_for(10 * 48_000, 48_000),
            audio_ctx_for(10 * 16_000, 16_000)
        );
    }
}
