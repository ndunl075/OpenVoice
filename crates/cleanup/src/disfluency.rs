//! Rule-based disfluency removal: fillers, stutters, and doubled words.
//!
//! This is the *fast* half of §2.4. The LLM pass in this crate's root is
//! the smart half, but it's off by default because it needs 516-677ms
//! against a 120ms budget (see `cleanup_enabled_by_default` in
//! `crates/daemon`). That left the common, boring disfluencies --
//! "um", "the the", "I-I-I" -- untouched, even though catching them
//! needs no understanding at all, just pattern matching.
//!
//! So: pure string work, no model, no allocation beyond the output, no
//! deadline to race. It runs on every utterance unconditionally because
//! it costs microseconds.
//!
//! **What it deliberately does not do.** It has no idea what you meant.
//! It won't fix grammar, restructure a sentence, or understand a spoken
//! self-correction ("send it Tuesday -- no, Wednesday"). Those need the
//! LLM pass, and no amount of rules gets there. The split is the point:
//! take the cheap deterministic wins for free, and leave genuine
//! language understanding to the model when it's affordable.

/// Filler words dropped when they stand alone as a whole token.
///
/// Deliberately short and unambiguous. Every entry here is a sound
/// people make while thinking, not a word that carries meaning --
/// which is why "ah", "oh", "well", "like" and "so" are *absent*
/// despite being common fillers: each has ordinary uses ("ah, I see",
/// "well water", "like this", "so cold") and silently deleting them
/// would change what the user said.
const FILLERS: &[&str] = &["um", "umm", "ummm", "uhm", "uh", "uhh", "uhhh", "er", "erm", "mmm"];

/// Words allowed to legitimately appear twice in a row, so the
/// repeat-collapsing rule leaves them alone.
///
/// English really does double these: "I had had enough", "the thing
/// that that person said". They're also common stutters, so this
/// exemption trades a missed cleanup for never corrupting a valid
/// sentence -- the right direction when the output goes straight into
/// the user's document with no review step.
const LEGITIMATE_DOUBLES: &[&str] = &["had", "that"];

/// Strips fillers, stutters, and accidental word repeats from raw ASR
/// text. Pure and instant; see the module docs for what it won't do.
pub fn strip_disfluencies(text: &str) -> String {
    let started_capitalized = text
        .chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(|c| c.is_uppercase());

    let mut kept: Vec<String> = Vec::new();
    for token in text.split_whitespace() {
        let (prefix, core, suffix) = split_affixes(token);
        let core = collapse_hyphen_stutter(core);

        if core.is_empty() || is_filler(&core) {
            // Dropping a filler must not eat its punctuation: "um," at a
            // clause boundary still leaves the comma doing real work, so
            // reattach it to whatever was kept last.
            if !suffix.is_empty() {
                if let Some(last) = kept.last_mut() {
                    if !last.ends_with(|c: char| !c.is_alphanumeric()) {
                        last.push_str(suffix);
                    }
                }
            }
            continue;
        }

        let rebuilt = format!("{prefix}{core}{suffix}");
        if repeats_previous(&kept, &core) {
            // Keep the later copy: it carries the punctuation that
            // follows the phrase ("the the," -> "the,").
            *kept.last_mut().expect("repeats_previous implies non-empty") = rebuilt;
        } else {
            kept.push(rebuilt);
        }
    }

    let mut out = kept.join(" ");
    if started_capitalized {
        capitalize_first_letter(&mut out);
    }
    out
}

/// Splits leading/trailing non-alphanumeric characters off a token, so
/// rules can compare the word itself while punctuation is preserved
/// verbatim in the output.
fn split_affixes(token: &str) -> (&str, String, &str) {
    let is_edge = |c: char| !c.is_alphanumeric();
    let start = token.find(|c: char| !is_edge(c)).unwrap_or(token.len());
    let end = token.rfind(|c: char| !is_edge(c)).map_or(start, |i| {
        i + token[i..].chars().next().map_or(1, char::len_utf8)
    });
    (&token[..start], token[start..end].to_string(), &token[end..])
}

fn normalized(word: &str) -> String {
    word.to_lowercase()
}

fn is_filler(core: &str) -> bool {
    FILLERS.contains(&normalized(core).as_str())
}

/// `"I-I-I"` -> `"I"`, `"wh-wh-what"` -> `"what"`.
///
/// Whisper transcribes stuttered starts as hyphen-joined fragments. Two
/// shapes are collapsed: the same fragment repeated, and a run of
/// fragments that are each a prefix of the final one (the false starts
/// of a word finally completed). Anything else -- real hyphenated words
/// like "well-known" or "state-of-the-art" -- is left alone, since its
/// parts are neither identical nor prefixes of the last.
fn collapse_hyphen_stutter(core: String) -> String {
    if !core.contains('-') {
        return core;
    }
    let parts: Vec<&str> = core.split('-').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 {
        return core;
    }
    let last = parts[parts.len() - 1];
    let last_norm = normalized(last);
    let all_stutter = parts[..parts.len() - 1]
        .iter()
        .all(|p| last_norm.starts_with(&normalized(p)));
    if all_stutter {
        last.to_string()
    } else {
        core
    }
}

fn repeats_previous(kept: &[String], core: &str) -> bool {
    let Some(previous) = kept.last() else {
        return false;
    };
    let core_norm = normalized(core);
    if LEGITIMATE_DOUBLES.contains(&core_norm.as_str()) {
        return false;
    }
    let (_, prev_core, _) = split_affixes(previous);
    normalized(&prev_core) == core_norm
}

fn capitalize_first_letter(s: &mut String) {
    let Some(pos) = s.find(|c: char| c.is_alphabetic()) else {
        return;
    };
    let ch = s[pos..].chars().next().expect("find returned a char boundary");
    if ch.is_uppercase() {
        return;
    }
    let upper: String = ch.to_uppercase().collect();
    s.replace_range(pos..pos + ch.len_utf8(), &upper);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_standalone_fillers() {
        assert_eq!(strip_disfluencies("um hello there"), "hello there");
        assert_eq!(strip_disfluencies("so uh we should go"), "so we should go");
    }

    #[test]
    fn collapses_a_doubled_word() {
        assert_eq!(strip_disfluencies("we should should go"), "we should go");
        assert_eq!(strip_disfluencies("the the cat"), "the cat");
    }

    #[test]
    fn collapsing_keeps_the_punctuation_that_followed() {
        assert_eq!(strip_disfluencies("wait wait, listen"), "wait, listen");
    }

    #[test]
    fn collapses_hyphenated_stutters() {
        assert_eq!(strip_disfluencies("I-I-I think so"), "I think so");
        assert_eq!(strip_disfluencies("wh-wh-what happened"), "what happened");
    }

    #[test]
    fn leaves_real_hyphenated_words_alone() {
        // Neither identical parts nor prefixes of the last -- not a stutter.
        let text = "a well-known state-of-the-art result";
        assert_eq!(strip_disfluencies(text), text);
    }

    #[test]
    fn leaves_legitimate_doubles_alone() {
        // Deleting one of these would change the sentence's meaning.
        assert_eq!(strip_disfluencies("I had had enough"), "I had had enough");
        let that = "the thing that that person said";
        assert_eq!(strip_disfluencies(that), that);
    }

    #[test]
    fn a_filler_does_not_swallow_its_punctuation() {
        // The comma is still doing real work at that clause boundary.
        assert_eq!(strip_disfluencies("okay um, let's go"), "okay, let's go");
    }

    #[test]
    fn does_not_treat_different_words_as_repeats() {
        assert_eq!(strip_disfluencies("very very nearly"), "very nearly");
        assert_eq!(strip_disfluencies("go to to the store"), "go to the store");
        assert_eq!(strip_disfluencies("cat sat"), "cat sat");
    }

    #[test]
    fn recapitalizes_only_when_the_original_was_capitalized() {
        // Real ASR output is capitalized, so dropping a leading filler
        // would otherwise leave the sentence starting lowercase.
        assert_eq!(strip_disfluencies("Um the cat"), "The cat");
        assert_eq!(strip_disfluencies("Um, hello there"), "Hello there");
        // ...but lowercase input stays lowercase; never invent a capital
        // the user didn't say.
        assert_eq!(strip_disfluencies("um the cat"), "the cat");
    }

    #[test]
    fn empty_and_filler_only_input_are_safe() {
        assert_eq!(strip_disfluencies(""), "");
        assert_eq!(strip_disfluencies("   "), "");
        assert_eq!(strip_disfluencies("um uh er"), "");
    }

    #[test]
    fn ordinary_text_passes_through_unchanged() {
        let clean = "Ship the release on Friday and tell the team.";
        assert_eq!(strip_disfluencies(clean), clean);
    }
}
