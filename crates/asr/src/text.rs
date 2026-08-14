//! Pure text post-processing for decoded segments, kept separate from the
//! `whisper-rs` FFI so it's unit-testable without a model loaded.

/// Joins whisper.cpp's per-segment text into one utterance: trims each
/// segment (whisper.cpp segments are typically space-prefixed), drops empty
/// segments, and joins the rest with single spaces.
pub fn join_segments<'a>(segments: impl IntoIterator<Item = &'a str>) -> String {
    segments
        .into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_and_trims_segments() {
        let segments = [" Hello", " world.", " "];
        assert_eq!(join_segments(segments), "Hello world.");
    }

    #[test]
    fn empty_input_is_empty_string() {
        let segments: Vec<&str> = vec![];
        assert_eq!(join_segments(segments), "");
    }

    #[test]
    fn single_segment_is_just_trimmed() {
        assert_eq!(join_segments([" only one "]), "only one");
    }

    #[test]
    fn drops_only_empty_and_whitespace_only_segments() {
        let segments = ["", "  ", "keep this", "\t"];
        assert_eq!(join_segments(segments), "keep this");
    }
}
