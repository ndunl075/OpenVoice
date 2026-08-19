//! Windows-only safety net for a real, reported bug: push-to-talk was
//! "feeling like a toggle" -- hold the chord, it starts; let go, it
//! doesn't stop until you press the keys *again*.
//!
//! Root cause isn't in [`crate::ChordDetector`]'s logic (that's pure and
//! thoroughly unit-tested) -- it's that `rdev::listen` on Windows is
//! built on a `WH_KEYBOARD_LL` global hook, and Windows will silently
//! disable a low-level hook mid-keystroke if its callback doesn't return
//! fast enough, dropping whatever event was in flight rather than
//! queuing it. For most keys that's an invisible missed keystroke; for
//! push-to-talk's `Released` transition specifically, a dropped
//! `KeyRelease` leaves the chord detector stuck believing both keys are
//! still down -- which is exactly "it turned into a toggle" from the
//! user's side, since the *next* physical press is the next event the
//! hook actually delivers.
//!
//! Rather than trying to make the hook itself more reliable (not
//! something this crate controls), this polls the OS's own real-time
//! idea of whether a key is physically down (`GetAsyncKeyState`, which
//! reflects hardware state directly, independent of whatever the hook's
//! event stream missed) as ground truth `daemon`'s watchdog thread can
//! cross-check the event-driven state against. See
//! `daemon::Engine::load`'s watchdog spawn for how it's used.

use rdev::Key;

/// Is `key` physically held down *right now*, per the OS -- not per
/// whatever `rdev::listen` events have (or haven't) delivered.
///
/// Returns `true` (assume still held) for any key [`to_vk_code`] doesn't
/// know how to map, and unconditionally on non-Windows targets: this
/// function only ever gets used to *force a release*, so the safe
/// default on "can't tell" is to not force one, not to guess.
#[cfg(windows)]
pub fn is_physically_down(key: Key) -> bool {
    let Some(vk) = to_vk_code(key) else {
        return true;
    };
    // High bit of GetAsyncKeyState's return value means "currently
    // down," and reflects real-time hardware state regardless of
    // whether any hook (ours or Windows') actually delivered an event
    // for it. See https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getasynckeystate
    unsafe { (windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState(vk as i32) as u16 & 0x8000) != 0 }
}

#[cfg(not(windows))]
pub fn is_physically_down(_key: Key) -> bool {
    // This module exists to patch a Windows-hook-specific limitation;
    // elsewhere, there's nothing more authoritative than the event
    // stream to check against, so trust it.
    true
}

/// Both keys of a push-to-talk chord, physically down right now.
pub fn both_physically_down(a: Key, b: Key) -> bool {
    is_physically_down(a) && is_physically_down(b)
}

/// Maps the subset of `rdev::Key` a realistic hotkey config would
/// actually use to a Win32 virtual-key code. Deliberately partial --
/// arrows/numpad/punctuation/etc. aren't listed, so they fall back to
/// [`is_physically_down`]'s safe "assume held" default rather than this
/// function guessing at a VK code for something it was never checked
/// against.
#[cfg(windows)]
fn to_vk_code(key: Key) -> Option<u16> {
    use windows::Win32::UI::Input::KeyboardAndMouse::*;
    Some(match key {
        Key::Alt => VK_MENU.0,
        Key::AltGr => VK_RMENU.0,
        Key::ControlLeft => VK_LCONTROL.0,
        Key::ControlRight => VK_RCONTROL.0,
        Key::ShiftLeft => VK_LSHIFT.0,
        Key::ShiftRight => VK_RSHIFT.0,
        Key::MetaLeft => VK_LWIN.0,
        Key::MetaRight => VK_RWIN.0,
        Key::CapsLock => VK_CAPITAL.0,
        Key::Tab => VK_TAB.0,
        Key::Space => VK_SPACE.0,
        Key::Return => VK_RETURN.0,
        Key::Escape => VK_ESCAPE.0,
        Key::Backspace => VK_BACK.0,
        Key::F1 => VK_F1.0,
        Key::F2 => VK_F2.0,
        Key::F3 => VK_F3.0,
        Key::F4 => VK_F4.0,
        Key::F5 => VK_F5.0,
        Key::F6 => VK_F6.0,
        Key::F7 => VK_F7.0,
        Key::F8 => VK_F8.0,
        Key::F9 => VK_F9.0,
        Key::F10 => VK_F10.0,
        Key::F11 => VK_F11.0,
        Key::F12 => VK_F12.0,
        // Win32 VK codes for letters/digits are literally their ASCII
        // codes (documented Win32 behavior) -- no named constants needed.
        Key::KeyA => b'A' as u16,
        Key::KeyB => b'B' as u16,
        Key::KeyC => b'C' as u16,
        Key::KeyD => b'D' as u16,
        Key::KeyE => b'E' as u16,
        Key::KeyF => b'F' as u16,
        Key::KeyG => b'G' as u16,
        Key::KeyH => b'H' as u16,
        Key::KeyI => b'I' as u16,
        Key::KeyJ => b'J' as u16,
        Key::KeyK => b'K' as u16,
        Key::KeyL => b'L' as u16,
        Key::KeyM => b'M' as u16,
        Key::KeyN => b'N' as u16,
        Key::KeyO => b'O' as u16,
        Key::KeyP => b'P' as u16,
        Key::KeyQ => b'Q' as u16,
        Key::KeyR => b'R' as u16,
        Key::KeyS => b'S' as u16,
        Key::KeyT => b'T' as u16,
        Key::KeyU => b'U' as u16,
        Key::KeyV => b'V' as u16,
        Key::KeyW => b'W' as u16,
        Key::KeyX => b'X' as u16,
        Key::KeyY => b'Y' as u16,
        Key::KeyZ => b'Z' as u16,
        Key::Num0 => b'0' as u16,
        Key::Num1 => b'1' as u16,
        Key::Num2 => b'2' as u16,
        Key::Num3 => b'3' as u16,
        Key::Num4 => b'4' as u16,
        Key::Num5 => b'5' as u16,
        Key::Num6 => b'6' as u16,
        Key::Num7 => b'7' as u16,
        Key::Num8 => b'8' as u16,
        Key::Num9 => b'9' as u16,
        _ => return None,
    })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn maps_the_default_push_to_talk_chord() {
        // MultiHotkeyConfig::default()'s actual keys -- if this mapping
        // regresses, the watchdog silently stops protecting the one
        // chord everyone actually uses.
        assert_eq!(to_vk_code(Key::ControlLeft), Some(0xA2)); // VK_LCONTROL
        assert_eq!(to_vk_code(Key::ShiftLeft), Some(0xA0)); // VK_LSHIFT
        assert_eq!(to_vk_code(Key::AltGr), Some(0xA5)); // VK_RMENU
    }

    #[test]
    fn letters_and_digits_map_to_their_ascii_codes() {
        assert_eq!(to_vk_code(Key::KeyA), Some(b'A' as u16));
        assert_eq!(to_vk_code(Key::Num5), Some(b'5' as u16));
    }

    #[test]
    fn unmapped_keys_return_none() {
        assert_eq!(to_vk_code(Key::LeftArrow), None);
        assert_eq!(to_vk_code(Key::Unknown(0)), None);
    }

    #[test]
    fn unmapped_key_is_assumed_physically_down() {
        // The safe default: never force a release for a key this
        // module can't actually check.
        assert!(is_physically_down(Key::LeftArrow));
    }
}
