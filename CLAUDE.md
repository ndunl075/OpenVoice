# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this project is

A local, on-device voice dictation engine written in Rust: hold a hotkey (or
toggle hands-free), speak, and the transcribed text is inserted at the cursor.
The product thesis is **latency plus privacy** — under ~200 ms from
end-of-speech to text at the cursor, and no audio ever leaves the machine.

Two binaries drive the same pipeline:

| Binary | Crate | What it is |
|---|---|---|
| `dictation-tray` | `crates/tray-app` | GUI: system tray icon + floating recording pill, no console window |
| `dictation-daemon` | `crates/daemon` | Console version, prints to stdout; headless/debugging use |

## Read this first: the architecture doc is the spec

[`dictation-architecture.md`](dictation-architecture.md) is the design document
this codebase implements, and it is treated as the source of truth. Its section
numbers are referenced *everywhere* — commit titles, module docs, inline
comments, crate READMEs:

```rust
/// §2.1: "capture the ~500 ms *before* the hotkey press. Solves the
/// universal 'I started talking a beat too early and lost my first word'
/// problem."
pub const PRE_ROLL_DURATION: Duration = Duration::from_millis(500);
```

**Follow this convention.** When you implement or change behavior that the doc
describes, cite the section (`§2.3`) and, where a constant or a design choice
comes straight from the doc, quote the sentence it comes from. When you're
about to change a magic number (120 ms cleanup deadline, 3 s windows with 0.5 s
overlap, 30 s ring buffer, 500 ms pre-roll), check the doc first — those
numbers are specified there, not arbitrary.

The doc's §4 build order (v0 → v1 → v2) is **fully implemented**. `README.md`'s
"Known gaps" section tracks what remains: GPU backend feature flags for
whisper.cpp (`cuda`/`metal`) are not enabled, so this build is CPU-only.

## Workspace layout

Cargo workspace, `resolver = "2"`, all crates at version `0.1.0` with
`version`/`edition`/`license` inherited from `[workspace.package]` (edition
2021).

```
crates/
  ring-buffer/   §2.1  Fixed-size in-memory circular audio buffer + pre-roll. ZERO dependencies.
  audio-input/   §2.1  cpal mic capture; downmix + resample to 16 kHz mono, feeds the ring buffer.
  hotkey/        §2.2  rdev global key capture; edge detection, two-key routing.
  vad/           §2.2  Silero VAD via ort (ONNX Runtime); frame chunking + endpoint state machine.
  asr/           §2.3  whisper-rs transcription; rolling-window policy, overlap merge, user dictionary.
  cleanup/       §2.4  llama.cpp (Qwen2.5-0.5B) disfluency cleanup behind a hard 120 ms deadline.
  inject/        §2.5  Clipboard-swap paste + per-character fallback; secure-field refusal.
  daemon/        §2    `Engine` library (the whole pipeline) + thin console binary.
  tray-app/      §3    egui/eframe pill window + tray-icon; drives the same `Engine`.
```

Dependency direction is strictly one-way: `tray-app` → `daemon` → every
component crate → (`ring-buffer` where audio is involved). Component crates
never depend on each other except through `ring-buffer`. **Don't introduce a
cross-dependency between component crates** — if two of them need to
coordinate, that coordination belongs in `daemon`.

## Pipeline data flow

```
mic ─cpal callback─▶ downmix_to_mono ─▶ linear_resample(→16 kHz) ─▶ RingBuffer (always on, ~30 s)
                                                                        │
hotkey (rdev thread) ──ControlEvent──▶ Engine::run loop ◀───────────────┘
                                            │  session = preroll(500 ms) ++ read_since(mark)
                                            ├─▶ SileroVad::process_frame (512-sample frames) ─▶ Endpointer
                                            ├─▶ WindowPolicy::next_window ─▶ Transcriber::transcribe ─▶ merge_overlap
                                            └─ on commit: final_window ─▶ CleanupModel::clean (120 ms race)
                                                                       ─▶ TextInjector::inject (once)
                                                                       ─▶ PipelineStatus channel ─▶ GUI
```

`crates/daemon/src/lib.rs` is the single orchestration point and by far the
most important file to read (~575 lines). Everything else is a component it
drives.

### The Engine contract

`daemon::Engine` is the seam between the pipeline and any front end:

- `Engine::load(print)` — loads models, opens the mic, installs the global
  hotkey hook. Takes a `impl FnMut(&str)` for progress lines so the console
  binary can pass `println!` and a GUI can route them elsewhere.
- `Engine::run(status_tx)` — **blocks** until `ControlEvent::Quit`. Run it on a
  background thread in a GUI (the OS event loop needs the main thread).
- `PipelineStatus` — structured state updates out (`Ready`, `Recording`,
  `Listening`, `Transcribing`, `Inserted`, `HeardNothing`, `HandsFreeOn/Off`,
  `Warning`). GUIs read this instead of scraping stdout.
- `ControlEvent` — events in, via `Engine::control_sender()`. The tray menu's
  "Toggle Hands-Free" sends the *same* `ControlEvent::HandsFreeTogglePressed`
  the physical AltGr key does.

**Preserve that last property.** There is deliberately no separate
"UI-triggered" code path; a menu action and a hotkey are provably identical
because they are the same event. Don't add a parallel API for the GUI.

## Non-negotiable design rules

These are enforced in code, and reviewers/commits treat them as load-bearing:

1. **Audio never touches disk or network.** `crates/ring-buffer/Cargo.toml` has
   an empty `[dependencies]` section on purpose — that's the constraint made
   verifiable rather than merely documented. Do not add a dependency to that
   crate. Nothing anywhere in the audio path makes a network call.
2. **Pick one string, insert once** (§2.4). Never insert raw text and patch it
   afterward. `CleanupModel::clean` returns `Option<String>`, not a stream or a
   correctable value; the timed-out generation is *abandoned*, never read late.
   `cleanup/src/deadline.rs` has a regression test guarding exactly this.
3. **Never inject into a secure field** (§2.5). `inject::is_focused_field_secure`
   is a best-effort Win32 `EM_GETPASSWORDCHAR` check — a floor, not a
   guarantee. It is permissive (`false`) on non-Windows so injection isn't
   blocked everywhere; if you implement a real check for another platform, do
   it there.
4. **Models are loaded once and stay resident.** Cold start is a latency line
   item only if you pay it more than once (§1). `Transcriber`, `SileroVad`, and
   `CleanupModel` are all constructed in `Engine::load` and reused.
5. **Optional models degrade, they don't fail.** ASR and VAD models missing =
   hard startup error. Cleanup model and user dictionary missing = logged and
   skipped, engine runs fine. Keep new optional features on the second pattern.

## Testing conventions

83 tests, all pure logic, colocated in `#[cfg(test)] mod tests` at the bottom
of the module they cover. **No test requires a model file, a microphone, or a
keyboard hook** — that's deliberate, and it's why the code is split the way it
is:

| Pure, tested | FFI/hardware, compile-verified only |
|---|---|
| `asr/window.rs`, `merge.rs`, `dictionary.rs`, `text.rs` | `asr/lib.rs` (`whisper-rs`) |
| `vad/framing.rs`, `endpoint.rs` | `vad/lib.rs` (`ort`) |
| `cleanup/deadline.rs`, `prompt.rs` | `cleanup/lib.rs` (`llama-cpp-2`) |
| `hotkey/edge.rs`, `multi.rs` | `hotkey`'s `listen*` (`rdev`) |
| `audio-input/convert.rs`, `resample.rs` | `audio-input/lib.rs` (`cpal`) |
| `tray-app/icon.rs` (pixel math) | `tray-app/app.rs`, `tray.rs` (`eframe`, `tray-icon`) |

**When adding a feature, extract the decidable logic into a pure module and
unit test it there.** Don't leave new policy logic inside an FFI call site.

Test names are full sentences describing the behavior, not
`test_foo`:

```rust
#[test]
fn read_since_saturates_when_session_overruns_capacity() { … }

#[test]
fn a_late_result_is_never_observed_after_giving_up() { … }
```

## Build, test, lint

```sh
cargo build --workspace
cargo test --workspace          # 83 tests
cargo clippy --workspace --all-targets -- -D warnings
```

Clippy warnings are errors in CI. Where a lint is knowingly suppressed, the
`#[allow(...)]` carries a comment explaining why (see
`cleanup/src/lib.rs`'s `explicit_counter_loop` allow).

### Native toolchain

Three crates compile a C/C++ inference engine from source, so a plain
`cargo build` is not enough:

- `cmake` on `PATH`, plus a C/C++ toolchain (MSVC on Windows).
- `LIBCLANG_PATH` pointing at a directory containing `libclang` — `bindgen`
  needs it for `whisper-rs-sys` and `ort`. CI sets
  `LIBCLANG_PATH: C:\Program Files\LLVM\bin`.
- `ort` downloads a prebuilt ONNX Runtime binary at build time (network
  access required); it does *not* need a system ONNX Runtime install.

Model weights are **not** in the repo (`.gitignore` excludes `/models/`,
`*.gguf`, `*.onnx`, `*.bin`). Fetch instructions are in the root `README.md`
and in `crates/asr`, `crates/vad`, `crates/cleanup` READMEs. Never commit a
model file.

### Platform reality — read before running commands

**Windows is the primary and only CI-tested target.** CI runs exclusively on
`windows-latest`. `inject` depends on the `windows` crate; `tray-app` is built
with `#![windows_subsystem = "windows"]`; secure-field detection is Win32-only.

In a Linux container (e.g. most agent sessions) the workspace does **not**
build as a whole. Measured on a typical agent container:

| Crate | Linux | Why |
|---|---|---|
| `ring-buffer` | ✅ 11 tests pass | No dependencies at all |
| `asr` | ✅ 27 tests pass | Needs `cmake` + `LIBCLANG_PATH`; multi-minute first build |
| `cleanup` | ✅ 6 tests pass | Same as `asr` |
| `vad` | ❌ | `ort` downloads a prebuilt ONNX Runtime from `cdn.pyke.io`; blocked by egress proxies |
| `hotkey` | ❌ | `rdev` → `x11` needs `libxi`/`libxtst` dev packages |
| `audio-input` | ❌ | `cpal` needs ALSA dev packages (`libasound2-dev`) |
| `daemon`, `tray-app` | ❌ | Depend on the above |

So roughly 44 of the 83 tests are reachable on Linux:

```sh
LIBCLANG_PATH=/usr/lib/llvm-18/lib cargo test -p ring-buffer -p asr -p cleanup
```

`cargo test --workspace` is **not** a meaningful local gate on Linux. Run what
you can, say clearly in your report what you could not verify, and let CI on
Windows be the real check. **Don't "fix" a Linux-only build failure by changing
dependencies or platform assumptions** — those failures are environmental.

## CI

`.github/workflows/ci.yml`: build → test → clippy, all with `--locked`, on
`windows-latest`, cached with `Swatinem/rust-cache`.

Because `--locked` is used, `Cargo.lock` must be committed and in sync whenever
dependencies change.

**Known quirk:** the workflow triggers on `pull_request` and on push to
`main`, but this repository's default branch is `master` — so the push trigger
never fires and CI effectively runs on PRs only. Worth knowing before you
conclude that a push "passed CI."

## Commit and PR conventions

Every feature so far landed as its own reviewed PR with green CI. Commit titles
follow:

```
<Imperative summary> (<architecture doc §>) (#<PR number>)
```

e.g. `Implement hands-free mode (§2.2, §4 v2) (#13)`,
`Stream windowed ASR decode, gated by VAD (§2.3, §4 v1) (#9)`.

Bodies in this repo are substantial and explain **why**, quoting the
architecture doc where a decision traces to it, calling out prerequisite
refactors separately from the feature, and noting what was actually verified
versus merely compiled. Match that depth; it's the house style.

Include the `Co-Authored-By:` trailer for AI-assisted commits.

## Gotchas

- `dictionary.txt` is gitignored user config; `dictionary.example.txt` is the
  tracked template. Same split-pattern applies to any future local config.
- Path resolution order is consistent: CLI arg (ASR model only) → env var →
  default under `models/`. Env vars: `DICTATION_MODEL_PATH`,
  `DICTATION_VAD_MODEL_PATH`, `DICTATION_CLEANUP_MODEL_PATH`,
  `DICTATION_DICTIONARY_PATH`.
- Silero's frame size is **512 samples** (32 ms), not the doc's nominal 30 ms —
  the model only accepts that exact size. `vad::next_frame_range` enforces it.
- `merge_overlap` is exact word matching, not a fuzzy aligner. It works because
  decoding is greedy and deterministic (fixed temperature, no fallback ladder);
  if you ever enable sampling or temperature fallback in `asr`, this stitching
  assumption breaks.
- `panic = "abort"` in the release profile — no unwinding, so don't rely on
  catching panics in release builds.
- Ring buffer lock failures use `.expect("ring buffer lock poisoned")`; the
  audio callback silently skips a chunk on lock failure instead of panicking on
  the realtime thread. Keep the audio callback panic-free.
- All UI assets are original and drawn in code (`tray-app/src/icon.rs`). The
  project name references Wispr Flow as a comparison point; **do not add any
  third-party logo, asset, or copied design file to this repo.**
