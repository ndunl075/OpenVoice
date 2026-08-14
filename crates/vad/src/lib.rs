//! Silero VAD endpointing via `ort` (ONNX Runtime).
//!
//! See `dictation-architecture.md` §2.2. Two jobs: trim silence before it
//! ever reaches the ASR encoder, and detect end-of-speech in a couple of
//! frames instead of a fixed timeout. The ONNX inference itself
//! ([`SileroVad`]) is inherently a model-dependent, hardware-facing piece
//! -- no model file ships in this repo, same as `asr`. The debounce logic
//! that turns raw per-frame probabilities into confirmed start/end events
//! ([`endpoint::Endpointer`]) is pure and fully unit tested; see that
//! module for why debouncing matters at all.

mod endpoint;
mod framing;

pub use endpoint::{EndpointConfig, EndpointEvent, Endpointer};
pub use framing::next_frame_range;

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// Silero VAD's native sample rate for this model export.
pub const SAMPLE_RATE_HZ: i64 = 16_000;
/// Silero VAD v4's native chunk size at 16kHz (32ms) -- the model is
/// trained on and expects exactly this many samples per inference step.
/// Close enough to the doc's nominal "30 ms frame" that no separate
/// resampling of the frame size is needed.
pub const FRAME_SAMPLES: usize = 512;

const STATE_LAYERS: usize = 2;
const STATE_DIM: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("expected a {expected}-sample frame, got {actual}")]
    WrongFrameSize { expected: usize, actual: usize },
    #[error("model produced no output probability")]
    EmptyOutput,
}

/// A loaded Silero VAD model plus its recurrent state (h/c), which persists
/// across calls to [`process_frame`](Self::process_frame) within a session
/// -- that's how the model gets temporal context from a stream of 32ms
/// chunks instead of judging each one in isolation.
pub struct SileroVad {
    session: Session,
    h: Vec<f32>,
    c: Vec<f32>,
}

impl SileroVad {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, VadError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path.as_ref())?;
        Ok(Self {
            session,
            h: vec![0.0; STATE_LAYERS * STATE_DIM],
            c: vec![0.0; STATE_LAYERS * STATE_DIM],
        })
    }

    /// Zeroes the recurrent state. Call at the start of a new listening
    /// session so history from a previous utterance doesn't bleed into the
    /// next one's probabilities.
    pub fn reset_state(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }

    /// Runs one inference step over exactly [`FRAME_SAMPLES`] of mono
    /// `f32` audio at [`SAMPLE_RATE_HZ`], returning the model's speech
    /// probability for that frame in `0.0..=1.0`. Feed the result into an
    /// [`Endpointer`] to turn a stream of these into start/end events.
    pub fn process_frame(&mut self, frame: &[f32]) -> Result<f32, VadError> {
        if frame.len() != FRAME_SAMPLES {
            return Err(VadError::WrongFrameSize {
                expected: FRAME_SAMPLES,
                actual: frame.len(),
            });
        }

        let input = Tensor::from_array(([1_i64, FRAME_SAMPLES as i64], frame.to_vec()))?;
        let sr = Tensor::from_array(([1_i64], vec![SAMPLE_RATE_HZ]))?;
        let h = Tensor::from_array((
            [STATE_LAYERS as i64, 1, STATE_DIM as i64],
            self.h.clone(),
        ))?;
        let c = Tensor::from_array((
            [STATE_LAYERS as i64, 1, STATE_DIM as i64],
            self.c.clone(),
        ))?;

        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "sr" => sr,
            "h" => h,
            "c" => c,
        ])?;

        let (_, probs) = outputs["output"].try_extract_tensor::<f32>()?;
        let probability = *probs.first().ok_or(VadError::EmptyOutput)?;

        let (_, hn) = outputs["hn"].try_extract_tensor::<f32>()?;
        self.h.copy_from_slice(hn);
        let (_, cn) = outputs["cn"].try_extract_tensor::<f32>()?;
        self.c.copy_from_slice(cn);

        Ok(probability)
    }
}
