# tray-app

GUI front end: a system tray icon plus a small floating "recording pill"
window, both driving the same `daemon::Engine` the console binary
(`crates/daemon`) uses. No terminal window on launch.

## What's here vs. what's original

This crate's visual language (a minimal floating pill that appears while
recording and briefly shows the result before fading, plus a tray icon)
follows the pattern popularized by dictation apps like Wispr Flow. The
**assets themselves are original**, not copied:

- [`icon.rs`](src/icon.rs) draws a simple circle-with-a-mic-glyph from
  scratch, in code -- there is no third-party logo file anywhere in this
  repo. The tray icon's background color changes with pipeline state
  (idle/recording/listening/thinking/done) so the mode is visible at a
  glance.
- The pill's colors, shape, and copy are original choices made for this
  project, not extracted from or matching any specific product's design
  files.

## Running

```sh
cargo run -p tray-app --release
```

Same model-fetching prerequisites as `crates/daemon` (see the root
README's "Running" section) -- this binary loads the exact same
`daemon::Engine`.

Model loading happens on the main thread before the tray icon/pill
appear (a few seconds, depending on hardware); if it fails, you'll get a
native message box rather than console output, since this binary has no
console in a normal launch.

## Controls

- Hold Right Ctrl: push-to-talk (same as the console binary).
- Tap AltGr: toggle hands-free mode, from the keyboard or the tray menu
  -- both send the identical control event, so there's no separate
  "UI-triggered" behavior to keep in sync with the hotkey path.
- Right-click the tray icon: "Toggle Hands-Free" / "Quit".

## Architecture note

`crates/daemon` was split into a library (`daemon::Engine`) and a thin
console binary specifically so this crate could exist without
duplicating the pipeline logic -- see `crates/daemon/src/lib.rs`'s module
docs for the `Engine::load` / `Engine::run` / `PipelineStatus` /
`ControlEvent` shape both front ends drive.
