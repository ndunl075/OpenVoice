//! Resident daemon entry point. Wires together the ring buffer, VAD, ASR,
//! cleanup pass, and text injector described in `dictation-architecture.md`.
//!
//! Each component is added to this binary incrementally; see the workspace
//! crates under `crates/` for the pieces as they land.

fn main() {
    println!("dictation-daemon: scaffold only, pipeline not wired up yet");
}
