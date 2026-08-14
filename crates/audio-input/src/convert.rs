//! Pure sample-format conversion helpers, kept separate from the `cpal`
//! device plumbing so they're unit-testable without real audio hardware.

/// Downmixes interleaved multi-channel `f32` samples to mono by averaging
/// the channels in each frame. A no-op copy when `channels == 1`.
pub fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels.max(1) as usize;
    if channels == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_passthrough_is_unchanged() {
        let input = [0.1, -0.2, 0.3];
        assert_eq!(downmix_to_mono(&input, 1), vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn stereo_averages_left_and_right() {
        // frame 1: L=1.0 R=3.0 -> 2.0; frame 2: L=-1.0 R=1.0 -> 0.0
        let input = [1.0, 3.0, -1.0, 1.0];
        assert_eq!(downmix_to_mono(&input, 2), vec![2.0, 0.0]);
    }

    #[test]
    fn drops_a_trailing_partial_frame() {
        // 5 samples at 2 channels: last sample has no pair, chunks_exact drops it.
        let input = [1.0, 1.0, 2.0, 2.0, 99.0];
        assert_eq!(downmix_to_mono(&input, 2), vec![1.0, 2.0]);
    }

    #[test]
    fn zero_channels_is_treated_as_mono() {
        let input = [0.5, 0.6];
        assert_eq!(downmix_to_mono(&input, 0), vec![0.5, 0.6]);
    }
}
