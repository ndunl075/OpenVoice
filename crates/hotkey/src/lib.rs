//! Global push-to-talk hotkey capture.
//!
//! See `dictation-architecture.md` §2.2. Push-to-talk is the v0/v1
//! interaction mode: hold the hotkey to record, release to commit. (Hands-
//! free -- VAD silence commits instead of a key release -- is a v2 addition
//! layered on top of the same event stream.)

mod edge;

pub use edge::EdgeDetector;
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
