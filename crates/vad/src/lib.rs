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
/// Silero VAD's native chunk size at 16kHz (32ms) -- the model is
/// trained on and expects exactly this many samples per inference step.
/// Close enough to the doc's nominal "30 ms frame" that no separate
/// resampling of the frame size is needed.
pub const FRAME_SAMPLES: usize = 512;

/// Shape of Silero **v5**'s single recurrent `state` tensor:
/// `[STATE_LAYERS, batch, STATE_DIM]`.
///
/// This crate was originally written against Silero **v4**, which took
/// two separate `h`/`c` LSTM tensors (`STATE_DIM` 64) and returned
/// `hn`/`cn`. The model the README tells you to download is v5, which
/// replaced all of that with one `state` in/`stateN` out and a width of
/// 128. The mismatch meant `session.run` returned
/// `InvalidArgument: Invalid input name: h` on **every single frame** --
/// and because `daemon`'s `feed_vad` only logs frame failures to stderr
/// (invisible in the GUI build, which has no console), the VAD silently
/// never ran at all: no speech was ever detected, streaming decode never
/// engaged, and every utterance fell back to one big decode at hotkey
/// release. Verified against the real model's reported signature --
/// inputs `["input", "state", "sr"]`, outputs `["output", "stateN"]` --
/// by `tests/real_speech_probe.rs`.
const STATE_LAYERS: usize = 2;
const STATE_DIM: usize = 128;

/// Samples of *previous* audio Silero v5 expects prepended to each chunk.
///
/// This is the subtle half of the v4->v5 change, and the one that caused
/// a silent wrong-answer rather than an error. v5's reference
/// implementation keeps a rolling context and feeds the model
/// `[context ++ chunk]` -- 64 + [`FRAME_SAMPLES`] = 576 samples at 16kHz
/// -- keeping the last 64 samples of each chunk as the next call's
/// context.
///
/// The ONNX graph declares `input` with a fully dynamic shape
/// (`[-1, -1]`), so feeding it a bare 512-sample chunk **does not
/// error**. It just returns a meaningless probability. Measured on the
/// bare chunk: silence 0.0006, clear speech 0.0006, loud noise 0.0011 --
/// i.e. the audio barely moved the output at all, which is what
/// distinguishes "wrong input layout" from "model disagrees with you".
const CONTEXT_SAMPLES: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),
    #[error("expected a {expected}-sample frame, got {actual}")]
    WrongFrameSize { expected: usize, actual: usize },
    #[error("model produced no output probability")]
    EmptyOutput,
}

/// A loaded Silero VAD model plus its recurrent state, which persists
/// across calls to [`process_frame`](Self::process_frame) within a session
/// -- that's how the model gets temporal context from a stream of 32ms
/// chunks instead of judging each one in isolation.
pub struct SileroVad {
    session: Session,
    /// v5's single `[STATE_LAYERS, 1, STATE_DIM]` recurrent state,
    /// flattened. See [`STATE_LAYERS`] for why this isn't `h`/`c`.
    state: Vec<f32>,
    /// Trailing [`CONTEXT_SAMPLES`] of the previous frame, prepended to
    /// the next one. See [`CONTEXT_SAMPLES`].
    context: Vec<f32>,
}

impl SileroVad {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, VadError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path.as_ref())?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_LAYERS * STATE_DIM],
            context: vec![0.0; CONTEXT_SAMPLES],
        })
    }

    /// Zeroes the recurrent state and audio context. Call at the start of
    /// a new listening session so history from a previous utterance
    /// doesn't bleed into the next one's probabilities.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
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

        // [context ++ frame], per CONTEXT_SAMPLES.
        let mut with_context = Vec::with_capacity(CONTEXT_SAMPLES + FRAME_SAMPLES);
        with_context.extend_from_slice(&self.context);
        with_context.extend_from_slice(frame);

        let input = Tensor::from_array((
            [1_i64, with_context.len() as i64],
            with_context.clone(),
        ))?;
        // `sr` is declared as a true scalar (shape []), not rank-1.
        let sr = Tensor::from_array(([0_i64; 0], vec![SAMPLE_RATE_HZ]))?;
        let state = Tensor::from_array((
            [STATE_LAYERS as i64, 1, STATE_DIM as i64],
            self.state.clone(),
        ))?;

        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "sr" => sr,
            "state" => state,
        ])?;

        let (_, probs) = outputs["output"].try_extract_tensor::<f32>()?;
        let probability = *probs.first().ok_or(VadError::EmptyOutput)?;

        let (_, next_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(next_state);

        // Carry this frame's tail as the next call's context.
        self.context
            .copy_from_slice(&with_context[with_context.len() - CONTEXT_SAMPLES..]);

        Ok(probability)
    }
}
