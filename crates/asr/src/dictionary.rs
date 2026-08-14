//! User dictionary -> `initial_prompt` biasing (§2.3: "Custom vocab: bias
//! decoding with an `initial_prompt` seeded from a user dictionary (names,
//! jargon, product names). Cheap, high perceived accuracy gain.")
//!
//! whisper.cpp's `initial_prompt` biases decoding toward text that *sounds
//! like* the prompt, without transcribing the prompt itself -- so a
//! natural-sounding sentence listing the vocabulary works better than a
//! bare comma-separated list. Parsing is split from file I/O so the format
//! is testable without touching disk.

use std::path::Path;

/// Parses a dictionary file's contents: one term per line, blank lines
/// and `#`-prefixed comment lines ignored, surrounding whitespace trimmed.
pub fn parse_dictionary(contents: &str) -> Vec<String> {
    contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Builds a whisper `initial_prompt` from dictionary terms. `None` if
/// there are no terms -- callers should leave `AsrConfig::initial_prompt`
/// unset in that case rather than biasing toward an empty list.
pub fn build_initial_prompt(terms: &[String]) -> Option<String> {
    if terms.is_empty() {
        return None;
    }
    Some(format!("Vocabulary that may appear: {}.", terms.join(", ")))
}

/// Reads and parses a dictionary file. A missing file is the expected,
/// non-fatal common case (the dictionary is an optional accuracy boost,
/// same spirit as the cleanup pass) -- callers should treat `Err` as "no
/// dictionary configured," not as a startup failure.
pub fn load_dictionary_file(path: impl AsRef<Path>) -> std::io::Result<Vec<String>> {
    let contents = std::fs::read_to_string(path)?;
    Ok(parse_dictionary(&contents))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_one_term_per_line() {
        let terms = parse_dictionary("Wispr\nQwen\nSilero");
        assert_eq!(terms, vec!["Wispr", "Qwen", "Silero"]);
    }

    #[test]
    fn ignores_blank_lines_and_comments() {
        let terms = parse_dictionary("Wispr\n\n# a comment\n  \nQwen\n#also a comment");
        assert_eq!(terms, vec!["Wispr", "Qwen"]);
    }

    #[test]
    fn trims_surrounding_whitespace() {
        let terms = parse_dictionary("  Wispr  \n\tQwen\t");
        assert_eq!(terms, vec!["Wispr", "Qwen"]);
    }

    #[test]
    fn empty_input_is_an_empty_list() {
        assert_eq!(parse_dictionary(""), Vec::<String>::new());
        assert_eq!(parse_dictionary("\n\n# only comments\n"), Vec::<String>::new());
    }

    #[test]
    fn empty_term_list_has_no_prompt() {
        assert_eq!(build_initial_prompt(&[]), None);
    }

    #[test]
    fn builds_a_natural_sentence_from_terms() {
        let terms = vec!["Wispr".to_string(), "Qwen2.5".to_string()];
        assert_eq!(
            build_initial_prompt(&terms),
            Some("Vocabulary that may appear: Wispr, Qwen2.5.".to_string())
        );
    }

    #[test]
    fn single_term_still_gets_a_full_sentence() {
        let terms = vec!["Silero".to_string()];
        assert_eq!(
            build_initial_prompt(&terms),
            Some("Vocabulary that may appear: Silero.".to_string())
        );
    }

    #[test]
    fn missing_file_is_a_plain_io_error_not_a_panic() {
        let result = load_dictionary_file("this/path/definitely/does/not/exist.txt");
        assert!(result.is_err());
    }
}
