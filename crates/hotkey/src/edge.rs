//! Press/release edge detection, kept separate from the `rdev` listener so
//! it's unit-testable without a real global keyboard hook.
//!
//! OS-level key-repeat means a held key fires `KeyPress` over and over.
//! Push-to-talk only cares about the transitions: the moment the key goes
//! down, and the moment it comes back up.

use rdev::{EventType, Key};

use crate::HotkeyEvent;

/// Tracks whether the configured key is currently held.
pub struct EdgeDetector {
    key: Key,
    is_down: bool,
}

impl EdgeDetector {
    pub fn new(key: Key) -> Self {
        Self {
            key,
            is_down: false,
        }
    }

    /// Feeds a raw input event. Returns `Some` only on a press/release
    /// *transition* of the configured key -- a repeated press while already
    /// down, a release while already up, or any other key, all yield
    /// `None`.
    pub fn handle(&mut self, event_type: &EventType) -> Option<HotkeyEvent> {
        match event_type {
            EventType::KeyPress(k) if *k == self.key => {
                if self.is_down {
                    None
                } else {
                    self.is_down = true;
                    Some(HotkeyEvent::Pressed)
                }
            }
            EventType::KeyRelease(k) if *k == self.key => {
                if self.is_down {
                    self.is_down = false;
                    Some(HotkeyEvent::Released)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn is_down(&self) -> bool {
        self.is_down
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn press_then_release_yields_one_transition_each() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        assert_eq!(
            d.handle(&EventType::KeyPress(Key::ControlRight)),
            Some(HotkeyEvent::Pressed)
        );
        assert_eq!(
            d.handle(&EventType::KeyRelease(Key::ControlRight)),
            Some(HotkeyEvent::Released)
        );
    }

    #[test]
    fn os_level_key_repeat_only_fires_once() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        assert_eq!(
            d.handle(&EventType::KeyPress(Key::ControlRight)),
            Some(HotkeyEvent::Pressed)
        );
        for _ in 0..10 {
            assert_eq!(d.handle(&EventType::KeyPress(Key::ControlRight)), None);
        }
        assert_eq!(
            d.handle(&EventType::KeyRelease(Key::ControlRight)),
            Some(HotkeyEvent::Released)
        );
    }

    #[test]
    fn release_while_already_up_is_ignored() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        assert_eq!(d.handle(&EventType::KeyRelease(Key::ControlRight)), None);
    }

    #[test]
    fn unrelated_keys_are_ignored() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        assert_eq!(d.handle(&EventType::KeyPress(Key::KeyA)), None);
        assert_eq!(d.handle(&EventType::KeyRelease(Key::KeyA)), None);
        assert!(!d.is_down());
    }

    #[test]
    fn non_keyboard_events_are_ignored() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        assert_eq!(d.handle(&EventType::MouseMove { x: 1.0, y: 2.0 }), None);
    }

    #[test]
    fn supports_multiple_press_release_cycles() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        for _ in 0..3 {
            assert_eq!(
                d.handle(&EventType::KeyPress(Key::ControlRight)),
                Some(HotkeyEvent::Pressed)
            );
            assert_eq!(
                d.handle(&EventType::KeyRelease(Key::ControlRight)),
                Some(HotkeyEvent::Released)
            );
        }
    }

    #[test]
    fn is_down_reflects_current_state() {
        let mut d = EdgeDetector::new(Key::ControlRight);
        assert!(!d.is_down());
        d.handle(&EventType::KeyPress(Key::ControlRight));
        assert!(d.is_down());
        d.handle(&EventType::KeyRelease(Key::ControlRight));
        assert!(!d.is_down());
    }
}
