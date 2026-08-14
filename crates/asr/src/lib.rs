//! Transcription via `whisper-rs` (whisper.cpp).
//!
//! See `dictation-architecture.md` §2.3. [`Transcriber::transcribe`]
//! decodes one chunk of audio at a time and is intentionally unopinionated
//! about *when* it's called: fed the whole utterance once at hotkey
//! release, it's v0's "batch on release" shape; fed successive rolling
//! windows while the user is still talking (via [`WindowPolicy`], stitched
//! back together with [`merge_overlap`]), it's v1's streaming shape --
//! the actual latency win (§1: "transcription should be nearly finished
//! before the user stops talking"). The daemon owns driving that loop;
//! see `crates/daemon` for the orchestration.
//!
//! Decode settings are fixed to what the doc calls out as the cheap,
//! high-value wins:
//! - `beam_size = 1` (greedy) -- the largest single speed win, modest WER cost
//! - no temperature fallback
//! - `no_context` (≈ `condition_on_previous_text = false`) -- avoids
//!   hallucination loops on short utterances
//! - `suppress_blank`, `no_timestamps` -- we only want plain text
//!
//! See [`README`](https://github.com/ndunl075/wispr-flow-clone/tree/main/crates/asr)
//! for how to fetch a model; none is checked into the repo.

mod merge;
mod text;
mod window;

pub use merge::merge_overlap;
pub use window::WindowPolicy;

use std::path::PathBuf;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub use whisper_rs::WhisperError;

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("model file not found at {0}: run the download step in crates/asr/README.md")]
    ModelNotFound(PathBuf),
    #[error("whisper.cpp error: {0}")]
    Whisper(#[from] WhisperError),
}

/// Everything the decoder needs to know about *how* to transcribe, as
/// opposed to *what audio* to transcribe.
#[derive(Debug, Clone)]
pub struct AsrConfig {
    /// Path to a ggml/whisper.cpp model file (e.g. `ggml-small.en-q5_1.bin`).
    pub model_path: PathBuf,
    /// `None` lets whisper.cpp auto-detect; for an English-only dictation
    /// tool, pinning this to `Some("en")` is both faster and more accurate.
    pub language: Option<String>,
    pub n_threads: i32,
    /// Seeds decoding to bias toward names/jargon/product names (§2.3,
    /// "custom vocab"). Wired up for real in the v2 user-dictionary commit;
    /// the field exists now so `AsrConfig` doesn't need to change shape
    /// later.
    pub initial_prompt: Option<String>,
}

impl AsrConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            language: Some("en".to_string()),
            n_threads: default_thread_count(),
            initial_prompt: None,
        }
    }
}

/// `min(available_parallelism, 4)`, clamped to at least 1. whisper.cpp's
/// own default is `min(4, hardware_concurrency)`; matching it explicitly
/// here means the daemon's thread budget is visible and overridable rather
/// than implicit in a C default.
pub fn default_thread_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(1, 4)
}

/// A loaded whisper.cpp model, ready to transcribe. Expensive to construct
/// (loads and pins the model in memory) and cheap to reuse -- the daemon
/// creates one of these at startup and keeps it for the process lifetime,
/// which is exactly what kills the "model cold start" latency in the naive
/// pipeline (§1).
pub struct Transcriber {
    ctx: WhisperContext,
    config: AsrConfig,
}

impl Transcriber {
    pub fn load(config: AsrConfig) -> Result<Self, AsrError> {
        if !config.model_path.is_file() {
            return Err(AsrError::ModelNotFound(config.model_path.clone()));
        }
        let ctx = WhisperContext::new_with_params(
            &config.model_path,
            WhisperContextParameters::default(),
        )?;
        Ok(Self { ctx, config })
    }

    /// Transcribes a full utterance of mono `f32` PCM audio at 16kHz.
    ///
    /// Batch mode: blocks until the whole clip is decoded. This is the
    /// v0 "decode on release" path; v1 replaces the caller of this with
    /// rolling windows decoded during speech instead.
    pub fn transcribe(&self, audio_16k_mono: &[f32]) -> Result<String, AsrError> {
        let mut state = self.ctx.create_state()?;
        let params = self.decode_params();
        state.full(params, audio_16k_mono)?;

        let n_segments = state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);
        for i in 0..n_segments {
            if let Some(segment) = state.get_segment(i) {
                segments.push(segment.to_str_lossy()?.into_owned());
            }
        }
        Ok(text::join_segments(segments.iter().map(String::as_str)))
    }

    fn decode_params(&self) -> FullParams<'_, '_> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.config.n_threads);
        params.set_translate(false);
        params.set_no_context(true); // condition_on_previous_text = false
        params.set_no_timestamps(true);
        params.set_suppress_blank(true);
        params.set_single_segment(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // No temperature-fallback decoding: fixed temperature, no retry ladder.
        params.set_temperature(0.0);
        params.set_temperature_inc(0.0);
        if let Some(lang) = self.config.language.as_deref() {
            params.set_language(Some(lang));
        }
        if let Some(prompt) = self.config.initial_prompt.as_deref() {
            params.set_initial_prompt(prompt);
        }
        params
    }
}
