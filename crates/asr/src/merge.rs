//! Stitches consecutive rolling-window transcripts back into one
//! utterance (§2.3: "3s windows with 0.5s overlap").
//!
//! Decoding runs with `no_timestamps = true` for speed (see the module
//! docs in `lib.rs`), so there's no sample-accurate boundary telling us
//! where the overlap actually falls in the *text*. Instead, this finds the
//! longest run of words at the end of what's already committed that
//! matches a run at the start of the new window's transcript, and appends
//! only what's new.
//!
//! **Matching is normalized (case- and punctuation-insensitive), and that
//! matters more than it sounds.** This used to compare words exactly, on
//! the reasoning that whisper.cpp's greedy deterministic decode gives the
//! same audio the same words. The words *are* the same -- but each window
//! is decoded independently (`no_context = true`), so whisper treats every
//! one as a fresh sentence and re-capitalizes and re-punctuates it. Real
//! observed failure: committed ended `"...what's going on"` while the next
//! window began `"What's going on?"`, which is the same three words and
//! matched on none of them. The overlap went undetected and the phrase was
//! appended verbatim, so a short utterance came out as
//! `"That's what's going on What's going on?"`.
//!
//! Comparison is normalized; the text that actually gets emitted is still
//! the original, unmodified window text.

/// Words compare equal if they match ignoring case and surrounding
/// punctuation. See the module docs for why exact matching was wrong.
fn normalized(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
        .to_lowercase()
}

/// Merges `next_window_text` into `committed`, deduplicating the overlap.
pub fn merge_overlap(committed: &str, next_window_text: &str) -> String {
    let committed_words: Vec<&str> = committed.split_whitespace().collect();
    let next_words: Vec<&str> = next_window_text.split_whitespace().collect();

    if committed_words.is_empty() {
        return next_words.join(" ");
    }
    if next_words.is_empty() {
        return committed_words.join(" ");
    }

    let committed_norm: Vec<String> = committed_words.iter().map(|w| normalized(w)).collect();
    let next_norm: Vec<String> = next_words.iter().map(|w| normalized(w)).collect();

    let max_overlap = committed_words.len().min(next_words.len());
    let overlap_len = (1..=max_overlap)
        .rev()
        .find(|&len| committed_norm[committed_norm.len() - len..] == next_norm[..len])
        .unwrap_or(0);

    let mut merged = committed_words.join(" ");
    let new_suffix = &next_words[overlap_len..];
    if !new_suffix.is_empty() {
        merged.push(' ');
        merged.push_str(&new_suffix.join(" "));
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_window_just_seeds_committed_text() {
        assert_eq!(merge_overlap("", "hello world"), "hello world");
    }

    #[test]
    fn appends_when_theres_no_overlap_at_all() {
        assert_eq!(
            merge_overlap("hello world", "goodbye now"),
            "hello world goodbye now"
        );
    }

    #[test]
    fn dedupes_a_multi_word_overlap() {
        assert_eq!(
            merge_overlap("the quick brown fox", "brown fox jumps over"),
            "the quick brown fox jumps over"
        );
    }

    #[test]
    fn dedupes_a_single_word_overlap() {
        assert_eq!(merge_overlap("hello world", "world peace"), "hello world peace");
    }

    #[test]
    fn identical_repeat_window_contributes_nothing_new() {
        assert_eq!(merge_overlap("hello world", "hello world"), "hello world");
    }

    #[test]
    fn empty_new_window_leaves_committed_untouched() {
        assert_eq!(merge_overlap("hello world", ""), "hello world");
        assert_eq!(merge_overlap("hello world", "   "), "hello world");
    }

    #[test]
    fn prefers_the_longest_matching_overlap() {
        // "a b a" vs "a b a b" -- the longest match is "a b a" (len 3), not
        // just the trailing "a" (len 1).
        assert_eq!(merge_overlap("x a b a", "a b a b"), "x a b a b");
    }

    #[test]
    fn is_idempotent_across_a_realistic_streaming_session() {
        let mut committed = String::new();
        for window in ["the quick brown", "brown fox jumps", "fox jumps over the lazy dog"] {
            committed = merge_overlap(&committed, window);
        }
        assert_eq!(committed, "the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn overlap_matches_despite_recapitalization_and_punctuation() {
        // The exact production failure: each window is decoded
        // independently, so whisper re-capitalizes and re-punctuates the
        // same words. Comparing exactly missed the overlap entirely and
        // produced "That's what's going on What's going on?".
        assert_eq!(
            merge_overlap("That's what's going on", "What's going on?"),
            "That's what's going on"
        );
    }

    #[test]
    fn emitted_text_keeps_the_original_words_not_the_normalized_ones() {
        // Normalization is only for *comparison* -- punctuation and case
        // in the actual output must survive untouched.
        assert_eq!(
            merge_overlap("Hello there", "There, world!"),
            "Hello there world!"
        );
    }

    #[test]
    fn a_trailing_period_does_not_hide_an_overlap() {
        assert_eq!(
            merge_overlap("we should ship it", "Ship it. Tomorrow"),
            "we should ship it Tomorrow"
        );
    }

    #[test]
    fn apostrophes_are_part_of_the_word_not_stripped_punctuation() {
        // "what's" must not normalize to "what" -- that would make
        // genuinely different words collide.
        assert_ne!(normalized("what's"), normalized("what"));
        assert_eq!(normalized("What's,"), normalized("what's"));
    }
}
