//! Global hotkey capture for both interaction modes.
//!
//! See `dictation-architecture.md` §2.2. Push-to-talk: hold a two-key
//! chord to record, release either key to commit -- deliberately two keys
//! rather than one common modifier, so it can't be tripped by accident
//! (see [`ChordDetector`]). Hands-free adds a second, independent
//! single-key toggle on top of the same edge-detection machinery
//! ([`listen_multi`]): tap it to start listening, and instead of a key
//! release, VAD silence commits each utterance (see `crates/daemon` for
//! where that decision actually gets made -- this crate only reports key
//! transitions, not audio state).

mod chord;
mod edge;
mod multi;

pub use chord::ChordDetector;
pub use edge::EdgeDetector;
pub use multi::{listen_multi, route, HotkeySlot, MultiHotkeyConfig};
pub use rdev::Key as HotkeyKey;

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
