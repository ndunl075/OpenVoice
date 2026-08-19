//! Routing for the daemon's two hotkey inputs: a two-key push-to-talk
//! chord, and a single-key hands-free toggle (§2.2: "Supports both
//! interaction modes: push-to-talk (release = commit) and hands-free (VAD
//! silence = commit)."). Each is tracked by its own detector so OS-level
//! key repeat is filtered independently for both, and holding push-to-talk
//! doesn't interfere with tapping the hands-free toggle or vice versa.
//!
//! Routing which detector a raw event belongs to is pulled out as
//! [`route`] so it's unit-testable without a real global hook -- same
//! pattern as [`ChordDetector`] and [`EdgeDetector`] themselves.

use rdev::{listen, Event, EventType};

use crate::{ChordDetector, EdgeDetector, HotkeyError, HotkeyEvent, HotkeyKey};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeySlot {
    PushToTalk,
    HandsFreeToggle,
}

#[derive(Debug, Clone, Copy)]
pub struct MultiHotkeyConfig {
    /// Both keys must be held together for push-to-talk -- a two-key
    /// chord rather than one common modifier key, so it can't be tripped
    /// by accident (e.g. bumping Right Ctrl while typing normally).
    pub push_to_talk_keys: (HotkeyKey, HotkeyKey),
    pub hands_free_toggle_key: HotkeyKey,
}

impl Default for MultiHotkeyConfig {
    /// Right Ctrl + Right Shift for push-to-talk: both reachable together
    /// under the right pinky without looking, and not a chord any common
    /// editor/OS shortcut uses. AltGr for the hands-free toggle --
    /// physically the right Alt key on most keyboard layouts, distinct
    /// from plain `Alt` in rdev.
    fn default() -> Self {
        Self {
            push_to_talk_keys: (HotkeyKey::ControlRight, HotkeyKey::ShiftRight),
            hands_free_toggle_key: HotkeyKey::AltGr,
        }
    }
}

/// Feeds one raw input event to both detectors and reports which slot (if
/// either) had a confirmed press/release transition. At most one of the
/// two can fire per event in normal configurations, but nothing stops a
/// caller from overlapping the chord and toggle keys, so this checks both
/// rather than assuming exclusivity.
pub fn route(
    ptt: &mut ChordDetector,
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
/// `on_event` on push-to-talk chord transitions and hands-free toggle key
/// transitions. Intended to run on a dedicated thread: like
/// `rdev::listen`, this only returns on error.
pub fn listen_multi(
    config: MultiHotkeyConfig,
    mut on_event: impl FnMut(HotkeySlot, HotkeyEvent) + Send + 'static,
) -> Result<(), HotkeyError> {
    let (key_a, key_b) = config.push_to_talk_keys;
    let mut ptt = ChordDetector::new(key_a, key_b);
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

    fn detectors() -> (ChordDetector, EdgeDetector) {
        (
            ChordDetector::new(Key::ControlRight, Key::ShiftRight),
            EdgeDetector::new(Key::AltGr),
        )
    }

    #[test]
    fn one_ptt_key_alone_does_not_route_as_pressed() {
        let (mut ptt, mut hf) = detectors();
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ControlRight)),
            None
        );
    }

    #[test]
    fn both_ptt_keys_together_route_to_push_to_talk_slot() {
        let (mut ptt, mut hf) = detectors();
        route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ControlRight));
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ShiftRight)),
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
        assert_eq!(route(&mut ptt, &mut hf, &EventType::KeyPress(Key::KeyA)), None);
    }

    #[test]
    fn the_two_detectors_track_independent_held_state() {
        let (mut ptt, mut hf) = detectors();
        // Hold push-to-talk down (both chord keys)...
        route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ControlRight));
        assert_eq!(
            route(&mut ptt, &mut hf, &EventType::KeyPress(Key::ShiftRight)),
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
        // Push-to-talk chord is still held.
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
