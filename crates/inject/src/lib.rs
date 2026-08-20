//! Text injection: gets dictated text from a `String` to "at the cursor."
//!
//! See `dictation-architecture.md` §2.5.
//!
//! Primary path: save the clipboard, swap in the dictated text, send a
//! synthetic paste chord, restore the clipboard shortly after. This is the
//! fastest and most universally compatible path (~15ms per the latency
//! table in §1) because it lets the target app do its own paste handling
//! instead of us simulating N keystrokes.
//!
//! Fallback: per-character synthetic keystrokes, for apps that block
//! programmatic paste (some terminals, secure fields in apps we can't
//! detect -- see [`secure_field`]).
//!
//! Never injects into a field flagged as secure input (best-effort check,
//! see [`secure_field::is_focused_field_secure`]).

mod secure_field;

pub use secure_field::{foreground_window_title, is_focused_field_secure};

use std::thread::sleep;
use std::time::Duration;

use enigo::{Direction, Enigo, Key, Keyboard, Settings};

/// How long to leave the dictated text sitting in the clipboard before
/// restoring whatever was there before. Needs to be long enough that the
/// target app has definitely finished reading the clipboard on paste.
pub const CLIPBOARD_RESTORE_DELAY: Duration = Duration::from_millis(50);

#[derive(Debug, thiserror::Error)]
pub enum InjectError {
    #[error("clipboard error: {0}")]
    Clipboard(#[from] arboard::Error),
    #[error("input simulation error: {0}")]
    Input(#[from] enigo::InputError),
    #[error("couldn't initialize input simulation: {0}")]
    Init(#[from] enigo::NewConError),
    #[error("refusing to inject into a field flagged as secure input")]
    SecureField,
}

pub struct TextInjector {
    enigo: Enigo,
}

impl TextInjector {
    pub fn new() -> Result<Self, InjectError> {
        Ok(Self {
            enigo: Enigo::new(&Settings::default())?,
        })
    }

    /// Inserts `text` at the cursor. Refuses outright if the focused field
    /// looks like a secure input. Tries the clipboard-swap paste path
    /// first; if sending the paste chord itself fails, falls back to
    /// per-character keystrokes rather than losing the utterance.
    ///
    /// Per §2.4's editing rule applied here too: this picks one string and
    /// inserts it once. It never inserts partial text and patches it.
    pub fn inject(&mut self, text: &str) -> Result<(), InjectError> {
        if text.is_empty() {
            return Ok(());
        }
        if is_focused_field_secure() {
            return Err(InjectError::SecureField);
        }
        match self.inject_via_paste(text) {
            Ok(()) => Ok(()),
            Err(_) => self.inject_via_keystrokes(text),
        }
    }

    /// Clipboard-swap + synthetic paste. Restores the previous clipboard
    /// contents afterward regardless of whether the paste itself
    /// succeeded, so a failed paste doesn't leave the user's clipboard
    /// silently clobbered.
    fn inject_via_paste(&mut self, text: &str) -> Result<(), InjectError> {
        let mut clipboard = arboard::Clipboard::new()?;
        let previous_text = clipboard.get_text().ok();

        clipboard.set_text(text.to_string())?;
        let paste_result = self.send_paste_chord();
        sleep(CLIPBOARD_RESTORE_DELAY);

        match previous_text {
            Some(prev) => {
                let _ = clipboard.set_text(prev);
            }
            None => {
                let _ = clipboard.clear();
            }
        }

        paste_result
    }

    /// Sends the paste chord, first clearing any modifier the user might
    /// still be physically holding.
    ///
    /// This matters because of how push-to-talk ends: the hotkey is a
    /// two-key chord (Ctrl+Shift by default) and an utterance commits when
    /// *either* key comes up. Release Ctrl a moment before Shift -- which
    /// is the normal way a hand leaves a chord -- and the paste fires while
    /// Shift is still down, so the OS sees Ctrl+**Shift**+V instead of
    /// Ctrl+V. Most apps don't paste on that, so the text silently never
    /// appeared while every layer here still reported success: enigo sent
    /// the keystrokes it was asked to, so `inject` returned `Ok`.
    ///
    /// Releasing them synthetically is safe from this app's perspective:
    /// `daemon` suppresses its own hotkey listener across injection, so
    /// these synthetic releases can't be mistaken for the user letting go.
    fn send_paste_chord(&mut self) -> Result<(), InjectError> {
        for stray in [Key::Shift, Key::Alt, Key::Meta] {
            // Best-effort: a platform that doesn't know one of these
            // shouldn't abort the paste.
            let _ = self.enigo.key(stray, Direction::Release);
        }

        let modifier = paste_modifier();
        self.enigo.key(modifier, Direction::Press)?;
        let result = self.enigo.key(Key::V, Direction::Click);
        self.enigo.key(modifier, Direction::Release)?;
        result?;
        Ok(())
    }

    /// Per-character fallback for apps that block programmatic paste.
    pub fn inject_via_keystrokes(&mut self, text: &str) -> Result<(), InjectError> {
        self.enigo.text(text)?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn paste_modifier() -> Key {
    Key::Meta
}

#[cfg(not(target_os = "macos"))]
fn paste_modifier() -> Key {
    Key::Control
}
