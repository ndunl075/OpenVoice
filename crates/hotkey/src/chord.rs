//! Two-key chord detection for push-to-talk: both keys have to be held
//! together, not just one -- a deliberate choice over a single common
//! modifier key (easy to trip accidentally, e.g. bumping Right Ctrl while
//! typing) since a two-key chord basically never happens by accident.
//!
//! Kept independent of `rdev`'s listener, same pattern as
//! [`crate::EdgeDetector`], so it's unit-testable without a real global
//! hook.

use rdev::{EventType, Key};

use crate::HotkeyEvent;

/// Tracks two keys' held state and reports a single `Pressed`/`Released`
/// transition for the *chord* -- i.e. "both keys down" as one logical
/// state, not two independent ones. `Pressed` fires the instant the
/// second key goes down (however the two were pressed); `Released` fires
/// the instant *either* key comes back up.
pub struct ChordDetector {
    key_a: Key,
    key_b: Key,
    a_down: bool,
    b_down: bool,
    /// Whether the chord is currently considered active (both were down
    /// as of the last confirmed transition) -- separate from `a_down &&
    /// b_down` so a repeated OS key-repeat event can't re-fire `Pressed`.
    chord_down: bool,
}

impl ChordDetector {
    pub fn new(key_a: Key, key_b: Key) -> Self {
        Self {
            key_a,
            key_b,
            a_down: false,
            b_down: false,
            chord_down: false,
        }
    }

    /// Feeds a raw input event. Returns `Some` only on a confirmed
    /// chord-level transition.
    pub fn handle(&mut self, event_type: &EventType) -> Option<HotkeyEvent> {
        match event_type {
            EventType::KeyPress(k) if *k == self.key_a => self.a_down = true,
            EventType::KeyRelease(k) if *k == self.key_a => self.a_down = false,
            EventType::KeyPress(k) if *k == self.key_b => self.b_down = true,
            EventType::KeyRelease(k) if *k == self.key_b => self.b_down = false,
            _ => return None,
        }

        let both_down = self.a_down && self.b_down;
        if both_down && !self.chord_down {
            self.chord_down = true;
            Some(HotkeyEvent::Pressed)
        } else if !both_down && self.chord_down {
            self.chord_down = false;
            Some(HotkeyEvent::Released)
        } else {
            None
        }
    }

    pub fn is_down(&self) -> bool {
        self.chord_down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> ChordDetector {
        ChordDetector::new(Key::ControlRight, Key::ShiftRight)
    }

    #[test]
    fn neither_key_alone_fires_pressed() {
        let mut d = detector();
        assert_eq!(d.handle(&EventType::KeyPress(Key::ControlRight)), None);
        assert!(!d.is_down());
        let mut d2 = detector();
        assert_eq!(d2.handle(&EventType::KeyPress(Key::ShiftRight)), None);
        assert!(!d2.is_down());
    }

    #[test]
    fn both_keys_down_fires_pressed_once() {
        let mut d = detector();
        assert_eq!(d.handle(&EventType::KeyPress(Key::ControlRight)), None);
        assert_eq!(
            d.handle(&EventType::KeyPress(Key::ShiftRight)),
            Some(HotkeyEvent::Pressed)
        );
        assert!(d.is_down());
    }

    #[test]
    fn order_of_the_two_keys_does_not_matter() {
        let mut d = detector();
        assert_eq!(d.handle(&EventType::KeyPress(Key::ShiftRight)), None);
        assert_eq!(
            d.handle(&EventType::KeyPress(Key::ControlRight)),
            Some(HotkeyEvent::Pressed)
        );
    }

    #[test]
    fn os_key_repeat_on_either_key_does_not_refire_pressed() {
        let mut d = detector();
        d.handle(&EventType::KeyPress(Key::ControlRight));
        d.handle(&EventType::KeyPress(Key::ShiftRight));
        for _ in 0..5 {
            assert_eq!(d.handle(&EventType::KeyPress(Key::ControlRight)), None);
            assert_eq!(d.handle(&EventType::KeyPress(Key::ShiftRight)), None);
        }
    }

    #[test]
    fn releasing_either_key_fires_released() {
        let mut d = detector();
        d.handle(&EventType::KeyPress(Key::ControlRight));
        d.handle(&EventType::KeyPress(Key::ShiftRight));
        assert_eq!(
            d.handle(&EventType::KeyRelease(Key::ControlRight)),
            Some(HotkeyEvent::Released)
        );
        assert!(!d.is_down());

        let mut d2 = detector();
        d2.handle(&EventType::KeyPress(Key::ControlRight));
        d2.handle(&EventType::KeyPress(Key::ShiftRight));
        assert_eq!(
            d2.handle(&EventType::KeyRelease(Key::ShiftRight)),
            Some(HotkeyEvent::Released)
        );
    }

    #[test]
    fn releasing_one_key_then_the_other_only_fires_released_once() {
        let mut d = detector();
        d.handle(&EventType::KeyPress(Key::ControlRight));
        d.handle(&EventType::KeyPress(Key::ShiftRight));
        assert_eq!(
            d.handle(&EventType::KeyRelease(Key::ControlRight)),
            Some(HotkeyEvent::Released)
        );
        // The second key releasing afterward is not a new transition --
        // the chord already ended when the first one let go.
        assert_eq!(d.handle(&EventType::KeyRelease(Key::ShiftRight)), None);
    }

    #[test]
    fn re_pressing_after_a_release_fires_pressed_again() {
        let mut d = detector();
        d.handle(&EventType::KeyPress(Key::ControlRight));
        d.handle(&EventType::KeyPress(Key::ShiftRight));
        d.handle(&EventType::KeyRelease(Key::ControlRight));
        d.handle(&EventType::KeyRelease(Key::ShiftRight));
        assert_eq!(d.handle(&EventType::KeyPress(Key::ControlRight)), None);
        assert_eq!(
            d.handle(&EventType::KeyPress(Key::ShiftRight)),
            Some(HotkeyEvent::Pressed)
        );
    }

    #[test]
    fn unrelated_keys_are_ignored() {
        let mut d = detector();
        assert_eq!(d.handle(&EventType::KeyPress(Key::KeyA)), None);
        assert_eq!(d.handle(&EventType::KeyRelease(Key::KeyA)), None);
        assert!(!d.is_down());
    }
}
