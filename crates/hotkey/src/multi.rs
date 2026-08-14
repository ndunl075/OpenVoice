//! Two-key routing for hands-free mode (§2.2: "Supports both interaction
//! modes: push-to-talk (release = commit) and hands-free (VAD silence =
//! commit)."). Push-to-talk and the hands-free toggle are separate
//! physical keys, each tracked by its own [`EdgeDetector`] so OS-level key
//! repeat is filtered independently for both.
//!
//! Routing which detector a raw event belongs to is pulled out as
//! [`route`] so it's unit-testable without a real global hook -- same
//! pattern as `EdgeDetector` itself.

use rdev::{listen, Event, EventType};

use crate::{EdgeDetector, HotkeyError, HotkeyEvent, HotkeyKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeySlot {
    PushToTalk,
    HandsFreeToggle,
}

#[derive(Debug, Clone, Copy)]
pub struct MultiHotkeyConfig {
    pub push_to_talk_key: HotkeyKey,
    pub hands_free_toggle_key: HotkeyKey,
}

impl Default for MultiHotkeyConfig {
    /// Right Ctrl for push-to-talk (see [`crate::HotkeyConfig`]'s default);
    /// AltGr for the hands-free toggle -- physically the right Alt key on
    /// most keyboard layouts, distinct from plain `Alt` in rdev, and not
    /// otherwise bound to anything dictation-adjacent.
    fn default() -> Self {
        Self {
            push_to_talk_key: HotkeyKey::ControlRight,
            hands_free_toggle_key: HotkeyKey::AltGr,
        }
    }
}

/// Feeds one raw input event to both detectors and reports which slot (if
/// either) had a confirmed press/release transition. At most one of the
/// two can fire per event in normal configurations (each detector ignores
/// events for keys other than its own), but nothing stops a caller from
/// pointing both slots at the same key, so this checks both rather than
/// assuming exclusivity.
pub fn route(
    ptt: &mut EdgeDetector,
    hands_free: &mut EdgeDetector,
    event_type: &EventType,
) -> Option<(HotkeySlot, HotkeyEvent)> {
    if let Some(event) = ptt.handle(event_type) {
        return Some((HotkeySlot::PushToTalk, event));
    }
    if let Some(event) = hands_free.handle(event_type) {
        return Some((HotkeySlot::HandsFreeToggle, event));
    }
    None
}

/// Blocks the calling thread, listening for global key events and invoking
/// `on_event` on press/release transitions of either configured key.
/// Intended to run on a dedicated thread: like `rdev::listen`, this only
/// returns on error.
pub fn listen_multi(
    config: MultiHotkeyConfig,
    mut on_event: impl FnMut(HotkeySlot, HotkeyEvent) + Send + 'static,
) -> Result<(), HotkeyError> {
    let mut ptt = EdgeDetector::new(config.push_to_talk_key);
    let mut hands_free = EdgeDetector::new(config.hands_free_toggle_key);
    listen(move |event: Event| {
        if let Some((slot, hk_event)) = route(&mut ptt, &mut hands_free, &event.event_type) {
            on_event(slot, hk_event);
        }
    })
    .map_err(HotkeyError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rdev::Key;

    fn detectors() -> (EdgeDetector, EdgeDetector) {
        (
            EdgeDetector::new(Key::ControlRight),
            EdgeDetector::new(Key::AltGr),
        )
    }

    #[test]
    fn routes_push_to_talk_key_to_push_to_talk_slot() {
        let (mut ptt, mut hf) = detectors();
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ControlRight)),
            Some((HotkeySlot::PushToTalk, HotkeyEvent::Pressed))
        );
    }

    #[test]
    fn routes_hands_free_key_to_hands_free_slot() {
        let (mut ptt, mut hf) = detectors();
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::AltGr)),
            Some((HotkeySlot::HandsFreeToggle, HotkeyEvent::Pressed))
        );
    }

    #[test]
    fn unrelated_key_routes_to_neither() {
        let (mut ptt, mut hf) = detectors();
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::KeyA)),
            None
        );
    }

    #[test]
    fn the_two_detectors_track_independent_held_state() {
        let (mut ptt, mut hf) = detectors();
        // Hold push-to-talk down...
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ControlRight)),
            Some((HotkeySlot::PushToTalk, HotkeyEvent::Pressed))
        );
        // ...tapping hands-free while it's held doesn't get swallowed or
        // confused with push-to-talk's held state.
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::AltGr)),
            Some((HotkeySlot::HandsFreeToggle, HotkeyEvent::Pressed))
        );
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyRelease(Key::AltGr)),
            Some((HotkeySlot::HandsFreeToggle, HotkeyEvent::Released))
        );
        // Push-to-talk is still held.
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ControlRight)),
            None,
            "still held -- OS repeat, not a new press"
        );
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyRelease(Key::ControlRight)),
            Some((HotkeySlot::PushToTalk, HotkeyEvent::Released))
        );
    }
}
