//! Stitches consecutive rolling-window transcripts back into one
//! utterance (§2.3: "3s windows with 0.5s overlap").
//!
//! Decoding runs with `no_timestamps = true` for speed (see the module
//! docs in `lib.rs`), so there's no sample-accurate boundary telling us
//! where the overlap actually falls in the *text*. Instead, this finds the
//! longest run of words at the end of what's already committed that
//! matches a run at the start of the new window's transcript, and appends
//! only what's new. Exact word-for-word matching only -- whisper.cpp's
//! greedy, deterministic decode (§2.3: fixed temperature, no fallback)
//! means the same audio in two overlapping windows decodes to the same
//! words often enough for this to work well in practice; it isn't a fuzzy
//! aligner.

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

    let max_overlap = committed_words.len().min(next_words.len());
    let overlap_len = (1..=max_overlap)
        .rev()
        .find(|&len| committed_words[committed_words.len() - len..] == next_words[..len])
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
}
