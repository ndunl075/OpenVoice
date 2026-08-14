//! Prompt construction, kept pure so the exact formatting is
//! unit-testable without a model loaded.
//!
//! Qwen2.5-Instruct models (§2.3: "Qwen2.5-0.5B-Instruct Q4") expect
//! ChatML-formatted input. Built by hand here rather than through a
//! chat-template helper, since the format is small, stable, and
//! well-documented, and hand-building it means this crate's prompt
//! behavior doesn't depend on GGUF metadata some quantizations omit.

const SYSTEM_PROMPT: &str = "You clean up dictated speech transcripts. Fix punctuation and \
capitalization, remove filler words like \"um\" and false starts, and otherwise preserve the \
speaker's wording and meaning exactly. Output only the cleaned text, nothing else -- no \
quotes, no commentary.";

/// Builds a ChatML prompt asking the model to clean up `raw_text`.
pub fn build_prompt(raw_text: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
         <|im_start|>user\n{raw_text}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_raw_text_in_chatml_turns() {
        let prompt = build_prompt("um so anyway i think we should ship it");
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.contains("<|im_start|>user\num so anyway i think we should ship it<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn instructs_output_only_no_commentary() {
        let prompt = build_prompt("hello");
        assert!(prompt.contains("Output only the cleaned text"));
    }

    #[test]
    fn does_not_lose_or_reorder_the_raw_text() {
        let raw = "the quick brown fox, um, jumps over";
        let prompt = build_prompt(raw);
        assert!(prompt.contains(raw));
    }
}
