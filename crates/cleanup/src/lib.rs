//! Deadlined disfluency-cleanup pass via `llama.cpp` (Qwen2.5-0.5B-Instruct).
//!
//! See `dictation-architecture.md` §2.4. Raw ASR output is disfluent: "um",
//! false starts, no punctuation. This runs a small resident LLM to fix
//! that up -- but only if it's fast:
//!
//! - **Hard deadline, 120ms.** If the model returns in time, use its
//!   output. If not, the caller falls back to the raw ASR text. The race
//!   itself ([`run_with_deadline`]) is pure and independent of llama.cpp;
//!   see [`deadline`] for why a timed-out generation is *abandoned*, not
//!   cancelled.
//! - **Pick one string, insert once.** §2.4 is explicit: "Do not insert
//!   raw text and then patch it afterward." [`CleanupModel::clean`]
//!   reflects that in its type -- it returns `Option<String>`, not a
//!   stream or a later-corrected value. The daemon decides once, with
//!   whatever answer is in hand at the deadline, and never revisits it.
//!
//! Model-dependent, hardware-facing like `asr`/`vad` -- no `.gguf` ships
//! in this repo; see `crates/cleanup/README.md`.

mod deadline;
mod disfluency;
mod prompt;

pub use deadline::{run_with_deadline, CancelToken};
pub use disfluency::strip_disfluencies;
pub use prompt::build_prompt;

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// §2.4: "Hard deadline: 120ms."
pub const CLEANUP_DEADLINE: Duration = Duration::from_millis(120);

/// Backstop on generated length. The deadline is the primary guard against
/// a runaway generation; this just makes sure a generation that somehow
/// keeps producing non-EOS tokens within the deadline can't ramble past
/// what a one-utterance cleanup should ever need.
const MAX_NEW_TOKENS: usize = 128;
const CONTEXT_TOKENS: u32 = 512;

#[derive(Debug, thiserror::Error)]
pub enum CleanupError {
    #[error("llama.cpp backend error: {0}")]
    Backend(#[from] llama_cpp_2::LlamaCppError),
    #[error("failed to load cleanup model: {0}")]
    ModelLoad(#[from] llama_cpp_2::LlamaModelLoadError),
    #[error("failed to create llama.cpp context: {0}")]
    ContextLoad(#[from] llama_cpp_2::LlamaContextLoadError),
    #[error("tokenization failed: {0}")]
    Tokenize(#[from] llama_cpp_2::StringToTokenError),
    #[error("detokenization failed: {0}")]
    Detokenize(#[from] llama_cpp_2::TokenToStringError),
    #[error("batch error: {0}")]
    Batch(#[from] llama_cpp_2::llama_batch::BatchAddError),
    #[error("decode error: {0}")]
    Decode(#[from] llama_cpp_2::DecodeError),
}

/// A loaded cleanup LLM, resident for the daemon's lifetime -- same
/// "load once, reuse forever" shape as `asr::Transcriber` and
/// `vad::SileroVad`, for the same reason (§1: model cold start is a
/// latency line item only if you pay it more than once).
pub struct CleanupModel {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    n_ctx: NonZeroU32,
}

impl CleanupModel {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, CleanupError> {
        let backend = LlamaBackend::init()?;
        let model = LlamaModel::load_from_file(&backend, model_path.as_ref(), &LlamaModelParams::default())?;
        Ok(Self {
            backend: Arc::new(backend),
            model: Arc::new(model),
            n_ctx: NonZeroU32::new(CONTEXT_TOKENS).expect("CONTEXT_TOKENS is nonzero"),
        })
    }

    /// Attempts to clean up `raw_text`, racing [`CLEANUP_DEADLINE`].
    /// `Some(cleaned)` if generation finished in time; `None` if it didn't
    /// (or failed) -- either way the caller should fall back to
    /// `raw_text` unmodified rather than wait any longer or patch things
    /// up after the fact.
    pub fn clean(&self, raw_text: &str) -> Option<String> {
        self.clean_with_deadline(raw_text, CLEANUP_DEADLINE)
    }

    /// [`clean`](Self::clean) with an explicit deadline, for testing and
    /// tuning without touching the doc's fixed 120ms constant.
    pub fn clean_with_deadline(&self, raw_text: &str, deadline: Duration) -> Option<String> {
        let backend = Arc::clone(&self.backend);
        let model = Arc::clone(&self.model);
        let n_ctx = self.n_ctx;
        let prompt = build_prompt(raw_text);

        run_with_deadline(deadline, move |cancel| {
            generate(&backend, &model, &prompt, n_ctx, &cancel)
        })
        .and_then(|result| result.ok())
    }
}

/// Runs one greedy generation to completion (or [`MAX_NEW_TOKENS`],
/// whichever comes first). Deliberately greedy, not sampled: this pass is
/// a deterministic cleanup step, not a creative one, and greedy decode
/// avoids needing to think about seeding across the abandon-on-timeout
/// boundary in [`run_with_deadline`].
fn generate(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    n_ctx: NonZeroU32,
    cancel: &CancelToken,
) -> Result<String, CleanupError> {
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
    let mut ctx = model.new_context(backend, ctx_params)?;

    let tokens = model.str_to_token(prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(n_ctx.get() as usize, 1);
    batch.add_sequence(&tokens, 0, false)?;
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::greedy();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur: i32 = tokens.len().try_into().unwrap_or(i32::MAX);
    let mut next_token = sampler.sample(&ctx, batch.n_tokens() - 1);

    // n_cur is the token's position in the KV cache, not a loop counter --
    // it must survive the loop unchanged in shape (clippy's suggested
    // rewrite would fold it into the range and lose that meaning).
    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..MAX_NEW_TOKENS {
        // The deadline has passed and this result will be discarded --
        // stop rather than keep decoding tokens nobody will read. This
        // is the common case, not an edge case: a full generation runs
        // 516-677ms against a 120ms deadline (see
        // `tests/abandoned_work_cost.rs`), so without this check every
        // utterance burns hundreds of milliseconds of inference purely
        // as heat. Whatever partial `output` we've accumulated is
        // thrown away by `run_with_deadline`'s caller anyway.
        if cancel.is_cancelled() {
            break;
        }
        if model.is_eog_token(next_token) {
            break;
        }
        // special=false: don't render control tokens (BOS/EOS/etc) into
        // the visible output text.
        output.push_str(&model.token_to_piece(next_token, &mut decoder, false, None)?);

        sampler.accept(next_token);
        batch.clear();
        batch.add(next_token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
        n_cur += 1;
        next_token = sampler.sample(&ctx, 0);
    }

    Ok(output.trim().to_string())
}
