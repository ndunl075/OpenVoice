//! Cheap linear-interpolation resampler.
//!
//! Not broadcast quality, but it's causal, allocation-light, and good
//! enough for VAD/ASR input — both of those models are already tolerant of
//! far worse artifacts than a linear resample introduces. Avoids pulling in
//! a heavyweight resampling dependency for a hot-path, latency-sensitive
//! component.

/// Resamples `input` from `from_hz` to `to_hz` via linear interpolation.
/// Returns the input unchanged (cloned) if the rates already match.
pub fn linear_resample(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_hz as f64 / from_hz as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = input.len() - 1;
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = (src_pos.floor() as usize).min(last);
        let frac = (src_pos - idx as f64) as f32;
        let a = input[idx];
        let b = input[(idx + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_rates_is_a_passthrough() {
        let input = [1.0, 2.0, 3.0];
        assert_eq!(linear_resample(&input, 16_000, 16_000), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(linear_resample(&[], 44_100, 16_000), Vec::<f32>::new());
    }

    #[test]
    fn downsamples_to_roughly_expected_length() {
        let input = vec![0.0f32; 44_100]; // 1s @ 44.1kHz
        let out = linear_resample(&input, 44_100, 16_000);
        // Should land within a sample or two of 16,000.
        assert!((out.len() as i64 - 16_000).abs() <= 2, "got {}", out.len());
    }

    #[test]
    fn upsamples_to_roughly_expected_length() {
        let input = vec![0.0f32; 16_000]; // 1s @ 16kHz
        let out = linear_resample(&input, 16_000, 48_000);
        assert!((out.len() as i64 - 48_000).abs() <= 2, "got {}", out.len());
    }

    #[test]
    fn interpolates_a_linear_ramp_exactly() {
        // A perfectly linear ramp should resample without error (up to fp noise),
        // since linear interpolation is exact for linear data -- except at the
        // very last output sample(s), which can land past the last interpolable
        // pair (idx, idx+1) and are clamped to the final input sample rather
        // than extrapolated past known data. That's the correct, causal
        // behavior for a streaming resampler: it never invents a sample it
        // hasn't seen yet.
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 0..9 @ "10Hz"
        let out = linear_resample(&input, 10, 20); // 2x upsample
        assert_eq!(out.len(), 20);
        for (i, &v) in out.iter().enumerate().take(19) {
            let expected = i as f32 * 0.5;
            assert!((v - expected).abs() < 1e-4, "index {i}: {v} vs {expected}");
        }
        // Last sample: src_pos = 9.5, but index 10 doesn't exist, so it
        // clamps to input[9] instead of the "true" 9.5.
        assert_eq!(out[19], 9.0);
    }
}
