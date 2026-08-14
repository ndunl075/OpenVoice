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

#[cfg(test)]
mod tests {
    use super::*;

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
