# tray-app

OpenVoice's GUI front end: a system tray icon plus a small floating
"recording pill" window, both driving the same `daemon::Engine` the
console binary (`crates/daemon`) uses. No terminal window on launch.

## What's here vs. what's original

This crate's *interaction pattern* (a minimal floating "bar" that appears
while recording, runs through a listening -> cleaning up -> final flow,
and fades after inserting text, plus a tray icon) follows the convention
dictation apps like Wispr Flow popularized -- the same way most apps in a
category share UI conventions without being copies of each other. The
**assets themselves are original**, not copied from anyone:

- [`icon.rs`](src/icon.rs) draws a simple circle-with-a-mic-glyph from
  scratch, in code -- there is no third-party logo file anywhere in this
  repo. The tray icon's background color changes with pipeline state
  (idle/recording/listening/cleaning up/done) so the mode is visible at a
  glance.
- The pill's colors, shape, and copy are original choices made for this
  project: a pale cream capsule (`icon::CREAM_BACKGROUND`) with one clear
  accent color per state -- coral while recording, lavender while
  listening, gold while cleaning up, sage on success -- and near-black
  text (`icon::DARK_TEXT`). Not extracted from any product's actual CSS
  or asset files, which we don't have; loosely follows the light,
  warm-cream surface this category of app tends to favor, with a
  lavender accent for the "actively listening" state, going by what's
  visible on public marketing pages.

## Running

```sh
cargo build -p tray-app --release
./target/release/dictation-tray.exe
```

**Don't use `cargo run -p tray-app` --** it's a reproduced Windows quirk:
`cargo run` exits almost instantly with code 0 and zero output for this
`windows_subsystem = "windows"` binary, even though the exact same
freshly-built `.exe` runs correctly every time when launched directly
(tray icon, models loaded, mic live, stays resident and responding).
Build once, then launch the `.exe` from `target/release/` directly --
after any code change, just re-run the build step first.

Same model-fetching prerequisites as `crates/daemon` (see the root
README's "Running" section) -- this binary loads the exact same
`daemon::Engine`.

Model loading happens on the main thread before the tray icon/pill
appear (a few seconds, depending on hardware); if it fails, you'll get a
native message box rather than console output, since this binary has no
console in a normal launch.

## Controls

- Hold Left Ctrl + Left Shift together: push-to-talk (same as the
  console binary). A two-key chord on purpose -- unlike a single modifier
  key, it can't be triggered by accident.
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
