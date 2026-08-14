//! Global hotkey capture for both interaction modes.
//!
//! See `dictation-architecture.md` §2.2. Push-to-talk ([`listen_push_to_talk`]):
//! hold the hotkey to record, release to commit -- the v0/v1 mode. Hands-
//! free adds a second, independent toggle key on top of the same
//! edge-detection machinery ([`listen_multi`]): tap it to start listening,
//! and instead of a key release, VAD silence commits each utterance (see
//! `crates/daemon` for where that decision actually gets made -- this
//! crate only reports key transitions, not audio state).

mod edge;
mod multi;

pub use edge::EdgeDetector;
pub use multi::{listen_multi, route, HotkeySlot, MultiHotkeyConfig};
pub use rdev::Key as HotkeyKey;

use rdev::{listen, Event};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    /// The hotkey transitioned from up to down.
    Pressed,
    /// The hotkey transitioned from down to up.
    Released,
}

#[derive(Debug, thiserror::Error)]
#[error("failed to install global hotkey listener: {0:?}")]
pub struct HotkeyError(rdev::ListenError);

#[derive(Debug, Clone, Copy)]
pub struct HotkeyConfig {
    pub key: HotkeyKey,
}

impl Default for HotkeyConfig {
    /// Right Ctrl: rarely bound to anything else, reachable without
    /// looking, doesn't collide with common editor/OS shortcuts.
    fn default() -> Self {
        Self {
            key: HotkeyKey::ControlRight,
        }
    }
}

/// Blocks the calling thread, listening for global key events and invoking
/// `on_event` on press/release *transitions* of `config.key`. Intended to
/// run on a dedicated thread: like `rdev::listen`, this only returns on
/// error.
pub fn listen_push_to_talk(
    config: HotkeyConfig,
    mut on_event: impl FnMut(HotkeyEvent) + Send + 'static,
) -> Result<(), HotkeyError> {
    let mut detector = EdgeDetector::new(config.key);
    listen(move |event: Event| {
        if let Some(hk_event) = detector.handle(&event.event_type) {
            on_event(hk_event);
        }
    })
    .map_err(HotkeyError)
}
