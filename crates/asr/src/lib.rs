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

mod audio_ctx;
mod dictionary;
mod merge;
mod text;
mod window;

pub use audio_ctx::audio_ctx_for;
pub use dictionary::{build_initial_prompt, load_dictionary_file, parse_dictionary};
pub use merge::merge_overlap;
pub use window::WindowPolicy;

/// Sample rate every audio buffer this crate touches is assumed to be at
/// (matches `ring_buffer::DEFAULT_SAMPLE_RATE_HZ` / `audio_input::TARGET_SAMPLE_RATE_HZ`).
const SAMPLE_RATE_HZ: u32 = 16_000;

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
    /// "custom vocab"). Build this from a user's dictionary file with
    /// [`load_dictionary_file`] + [`build_initial_prompt`].
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

/// `available_parallelism / 2`, clamped to `[1, 8]`.
///
/// This used to just match whisper.cpp's own upstream default of
/// `min(4, hardware_concurrency)` -- reasonable-sounding, but never
/// actually benchmarked on this machine. It wasn't: real measurement
/// (`crates/asr/tests/real_time_factor.rs`, run with
/// `ASR_BENCH_MODEL_A=... cargo test --release -p asr --test
/// real_time_factor -- --ignored --nocapture`) on a 16-logical-core
/// machine found 4 threads decoding at ~1.1-1.2x real-time (i.e.
/// *slower* than the audio itself -- the streaming pipeline can't keep
/// up), 8 threads at ~0.8x (comfortably real-time), and going all the
/// way to 16 threads catastrophically regressing to 10-25x real-time --
/// almost certainly thread-pool/scheduling contention once you cross
/// from physical into hyperthreaded cores. So: more isn't always
/// better, but 4 was leaving real, free speed on the table. Halving
/// `available_parallelism` is a rough physical-core estimate on typical
/// 2-way-SMT hardware without hardcoding a number that's wrong on a
/// machine with a different core count; the `[1, 8]` clamp keeps it
/// from ever wandering back into the regime that regressed at 16.
pub fn default_thread_count() -> i32 {
    let available = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
    (available / 2).clamp(1, 8)
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
        let params = self.decode_params(audio_16k_mono.len());
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

    fn decode_params(&self, sample_count: usize) -> FullParams<'_, '_> {
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
        // The single biggest speed lever here: scope the encoder's context
        // to how much audio actually arrived instead of paying for a full
        // 30s context every time (whisper.cpp's `audio_ctx` default of 0
        // means "full 30s" regardless of input length). See audio_ctx.rs
        // for the ~11.7x measurement behind this.
        params.set_audio_ctx(audio_ctx::audio_ctx_for(sample_count, SAMPLE_RATE_HZ));
        // Bug fix: this used to also call `set_temperature_inc(0.0)`,
        // which fully disables whisper.cpp's temperature-fallback retry
        // ladder -- the mechanism that re-decodes at a higher temperature
        // when the greedy pass's own entropy/logprob checks say the
        // result looks bad, instead of just accepting whatever garbage
        // greedy produced. That's *the* built-in defense against
        // hallucinated/repeated-word output on short, context-free
        // (`no_context = true`) clips -- exactly what every rolling
        // window here is. Turning it off traded "no unpredictable
        // decode-time retries" for "confidently emits nonsense on
        // ambiguous audio," which is a bad trade for a dictation tool.
        // `temperature_inc = 0.2` restores whisper.cpp's own upstream
        // default (entropy_thold/logprob_thold are already at their
        // sane defaults from `whisper_full_default_params` above; we've
        // never overridden those). The retry only fires on windows the
        // model itself flags as low-confidence, so this shouldn't cost
        // anything on the common case where the first pass was fine.
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        if let Some(lang) = self.config.language.as_deref() {
            params.set_language(Some(lang));
        }
        if let Some(prompt) = self.config.initial_prompt.as_deref() {
            params.set_initial_prompt(prompt);
        }
        params
    }
}
