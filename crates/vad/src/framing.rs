//! Chunks a growing audio stream into exactly [`crate::FRAME_SAMPLES`]
//! pieces for [`crate::SileroVad::process_frame`], kept pure so the
//! chunking logic is testable independent of any ONNX inference.

use crate::FRAME_SAMPLES;

/// The `[start, end)` range of the next full VAD frame ready to process,
/// given how many samples have already been fed to the VAD
/// (`fed_samples`) and how many are currently available
/// (`available_samples`). Returns `None` if less than a full frame's worth
/// of new audio has arrived yet -- the caller should just wait for more.
///
/// Callers advance by exactly `FRAME_SAMPLES` each time (rather than
/// consuming all available audio in one variable-sized chunk) because
/// Silero VAD's ONNX graph is trained on, and only accepts, this fixed
/// frame size.
pub fn next_frame_range(fed_samples: usize, available_samples: usize) -> Option<(usize, usize)> {
    let end = fed_samples + FRAME_SAMPLES;
    (end <= available_samples).then_some((fed_samples, end))
}

/// Where speech actually began, given the endpointer only *confirmed* it
/// at `confirmed_end_sample` after `start_frames` consecutive
/// above-threshold frames.
///
/// This is the "trim silence so the encoder never wastes compute on dead
/// air" half of §2.2, and it fixes a real user-visible bug, not just
/// wasted compute: §2.1's pre-roll deliberately prepends ~500ms of audio
/// from *before* the hotkey press, which is usually near-silence -- and
/// whisper reliably hallucinates short filler out of silence ("Okay.",
/// "Yeah", "Thank you."). Those turned up as phantom words at the start
/// of transcripts. Decoding from real speech onward removes the input
/// that produced them.
///
/// Two corrections are applied, both backwards in time, because clipping
/// the start of a word is worse than decoding a little extra silence:
///
/// - **confirmation lag**: the endpointer needs `start_frames` frames of
///   evidence before it says "speech", so speech began at least that far
///   back.
/// - **`lead_in_samples`**: additional slack for the attack of the first
///   phoneme, which is often quiet enough not to have crossed the
///   threshold itself (plosives and fricatives especially).
///
/// Saturating throughout: the estimate simply floors at 0 when speech
/// starts near the very beginning of the buffer.
pub fn speech_start_estimate(
    confirmed_end_sample: usize,
    start_frames: u32,
    lead_in_samples: usize,
) -> usize {
    confirmed_end_sample
        .saturating_sub(start_frames as usize * FRAME_SAMPLES)
        .saturating_sub(lead_in_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_start_backs_up_past_the_confirmation_lag_and_lead_in() {
        // Confirmed at sample 10,000 after 3 frames of evidence, with a
        // 1,000-sample lead-in: 10_000 - 3*512 - 1_000.
        assert_eq!(
            speech_start_estimate(10_000, 3, 1_000),
            10_000 - 3 * FRAME_SAMPLES - 1_000
        );
    }

    #[test]
    fn speech_start_never_goes_negative() {
        // Speech confirmed almost immediately -- there's nothing before
        // the buffer to back into.
        assert_eq!(speech_start_estimate(100, 3, 1_000), 0);
        assert_eq!(speech_start_estimate(0, 3, 1_000), 0);
    }

    #[test]
    fn speech_start_is_never_after_the_confirmation_point() {
        // The estimate must only ever move backwards; returning a later
        // sample would clip off the very speech that triggered it.
        for confirmed in [0, 500, 5_000, 50_000] {
            assert!(speech_start_estimate(confirmed, 3, 1_000) <= confirmed);
        }
    }

    #[test]
    fn no_frame_until_a_full_frame_is_available() {
        assert_eq!(next_frame_range(0, FRAME_SAMPLES - 1), None);
        assert_eq!(next_frame_range(0, FRAME_SAMPLES), Some((0, FRAME_SAMPLES)));
    }

    #[test]
    fn advances_by_exactly_one_frame_at_a_time() {
        assert_eq!(
            next_frame_range(FRAME_SAMPLES, FRAME_SAMPLES),
            None,
            "no second frame's worth of new audio yet"
        );
        assert_eq!(
            next_frame_range(FRAME_SAMPLES, FRAME_SAMPLES * 2),
            Some((FRAME_SAMPLES, FRAME_SAMPLES * 2))
        );
    }

    #[test]
    fn leftover_partial_audio_is_never_returned_as_a_frame() {
        // 1.5 frames available, already fed one frame: the trailing half
        // isn't a full frame, so nothing's ready.
        assert_eq!(
            next_frame_range(FRAME_SAMPLES, FRAME_SAMPLES + FRAME_SAMPLES / 2),
            None
        );
    }

    #[test]
    fn consecutive_calls_walk_through_a_stream_without_gaps_or_overlap() {
        let available = FRAME_SAMPLES * 3 + 10; // three full frames, plus a leftover partial
        let mut fed = 0;
        let mut frames = Vec::new();
        while let Some((start, end)) = next_frame_range(fed, available) {
            frames.push((start, end));
            fed = end;
        }
        assert_eq!(
            frames,
            vec![
                (0, FRAME_SAMPLES),
                (FRAME_SAMPLES, FRAME_SAMPLES * 2),
                (FRAME_SAMPLES * 2, FRAME_SAMPLES * 3),
            ]
        );
    }
}
