# Local Dictation Engine

On-device voice dictation: text appears at the cursor in well under 200 ms
after you stop talking, and no audio ever leaves the machine. Full design
rationale lives in [`dictation-architecture.md`](dictation-architecture.md) —
this README tracks what's actually built.

## Status

The architecture doc's full build order is implemented: **v0 → v1 → v2**,
each merged as its own reviewed PR with a green CI run.

- [x] **v0 — demo pipeline:** ring buffer, push-to-talk hotkey, batch Whisper
      on release, clipboard-paste injection.
- [x] **v1 — the real win:** Silero VAD endpointing, streaming windowed decode.
- [x] **v2 — quality:** deadlined cleanup LLM pass, user dictionary, pre-roll
      capture, hands-free mode.

That's every item §4's build order calls for. See "Known gaps" below for
what's in the doc but *outside* that checklist (§3's stack table has a
couple of entries §4 never actually requires) and hasn't been built.

## Running

```sh
# 1. Fetch models (see crates/asr, crates/vad, crates/cleanup READMEs)
mkdir -p models
curl -L -o models/ggml-small.en-q5_1.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin
curl -L -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# 2. Build, then run the built .exe directly -- see the note below on
#    why `cargo run` specifically doesn't work for tray-app.
cargo build -p tray-app --release
./target/release/dictation-tray.exe

# ...or the console version, if you'd rather see raw log output
cargo run -p daemon --release
```

> **`cargo run -p tray-app` looks like it does nothing -- use the two-step
> build-then-run above instead.** This is a real, reproduced quirk of
> `cargo run` wrapping a `windows_subsystem = "windows"` binary on
> Windows: `cargo run` exits ~instantly with code 0 and no output, while
> the exact same freshly-built `dictation-tray.exe` runs correctly (tray
> icon, models loaded, mic live, stays resident) every time when launched
> directly. `daemon` (console subsystem) doesn't have this problem --
> `cargo run -p daemon` is fine. If you rebuild `tray-app` after changing
> its code, re-run `cargo build -p tray-app --release` and launch the
> `.exe` again the same way.

Hold Right Ctrl to dictate, release to insert the transcribed text at the
cursor -- or tap AltGr to switch to hands-free mode (see below). As of v1,
transcription streams continuously while the key is held
(rolling 3s windows, §2.3) instead of waiting for release -- only the
trailing partial window is left to decode at that point. The ASR model path
defaults to `models/ggml-small.en-q5_1.bin`; override it with a CLI arg
(`cargo run -p daemon --release -- path/to/model.bin`) or the
`DICTATION_MODEL_PATH` env var. The VAD model path defaults to
`models/silero_vad.onnx`; override with `DICTATION_VAD_MODEL_PATH`. The
cleanup model (§2.4) is optional -- if `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
(or `DICTATION_CLEANUP_MODEL_PATH`) isn't found, the daemon logs that and
runs without it rather than refusing to start; every utterance falls back
to raw ASR text.

**User dictionary (§2.3 "custom vocab"):** copy [`dictionary.example.txt`](dictionary.example.txt)
to `dictionary.txt` (or point `DICTATION_DICTIONARY_PATH` at your own file)
and list names/jargon/product names, one per line, to bias ASR decoding
toward them. Also optional -- no file means no bias, same as before this
existed.

**Pre-roll (§2.1):** every recording session includes the ~500ms of audio
from just before the hotkey went down, pulled straight from the always-on
ring buffer. Solves "I started talking a beat too early and lost my first
word" -- something a cloud tool structurally can't do, since it isn't
listening until you've already pressed the key.

**Hands-free mode (§2.2, §4 v2):** tap AltGr (physically Right Alt on most
layouts) to toggle hands-free listening on. Instead of holding a key,
Silero VAD's confirmed end-of-speech commits each utterance -- and the
daemon immediately starts listening for the next one, so a whole
dictation session can run without touching the keyboard between
utterances. Tap AltGr again to stop. Push-to-talk (Right Ctrl) and
hands-free are mutually exclusive: whichever session is active, the other
mode's key is ignored until it ends.

**Recording indicator:** `tray-app` (see [`crates/tray-app`](crates/tray-app))
gives you a system tray icon -- an original mic glyph, not a copy of any
product's logo, whose background color reflects pipeline state at a
glance -- plus a small floating pill near the bottom of the screen that
appears while recording/listening/thinking and briefly shows the result
before fading. The architecture doc calls a visible indicator out as
necessary before an always-on mic is trustworthy to ship publicly (see
"Honest risks" in `dictation-architecture.md`); this is that. The
console binary (`daemon`) still exists and still just prints to stdout,
for headless/debugging use -- `tray-app` is the one with an actual UI.

## Privacy: the always-on buffer

The daemon keeps a rolling ~30s audio buffer in memory so the mic device
never has to be re-opened on hotkey press (see §2.1 of the architecture
doc). This is the biggest single latency win in the whole design, and also
the thing an on-device product cannot be sloppy about:

- The buffer lives in a fixed-size `Vec<f32>` on the heap. It is **never
  written to disk**, and old samples are overwritten in place as new audio
  arrives — nothing is retained past the ring's ~30s window.
- Audio only leaves the ring buffer when a hotkey press or VAD trigger pulls
  a slice out for transcription. Transcription is 100% local; no network
  calls exist anywhere in the audio path.
- A visible recording indicator is required before "always-on" ships
  publicly (see "Recording indicator" above) -- `tray-app`'s tray icon
  and floating pill are that indicator.
- This is enforced in code, not just documented: [`crates/ring-buffer`](crates/ring-buffer)
  has no file I/O and no network dependency at all — check its `Cargo.toml`.

## Workspace layout

| Crate | Architecture doc section | Purpose |
|---|---|---|
| [`ring-buffer`](crates/ring-buffer) | §2.1 | Fixed-size in-memory circular audio buffer, pre-roll support |
| [`audio-input`](crates/audio-input) | §2.1 | `cpal` mic capture feeding the ring buffer |
| [`hotkey`](crates/hotkey) | §2.2 | Global push-to-talk / hands-free hotkey capture |
| [`vad`](crates/vad) | §2.2 | Silero VAD endpointing via `ort` |
| [`asr`](crates/asr) | §2.3 | Streaming/batch transcription via `whisper-rs` |
| [`cleanup`](crates/cleanup) | §2.4 | Deadlined disfluency cleanup via `llama.cpp` |
| [`inject`](crates/inject) | §2.5 | Clipboard-swap paste, per-character fallback |
| [`daemon`](crates/daemon) | §2 | The engine as a library (`Engine`) + a thin console binary |
| [`tray-app`](crates/tray-app) | §3 ("Tray + minimal overlay") | System tray icon + floating recording pill GUI, drives the same `Engine` |

## Known gaps vs. the architecture doc

§4's build order (the actual required checklist) is fully implemented,
and so is §3's "Tray + minimal overlay" ([`tray-app`](crates/tray-app)).
One thing §3 mentions is still not built:

- **GPU backends.** §2.3 mentions Metal/CUDA backends for whisper.cpp;
  `asr`'s `Cargo.toml` doesn't enable whisper-rs's `cuda`/`metal` feature
  flags, so this build is CPU-only (whisper.cpp's own AVX2 auto-detection
  still applies -- that's the "CPU/AVX2 fallback" leg of §2.3's backend
  list, just not the GPU-accelerated ones). Enabling `cuda` needs a CUDA
  toolkit on the build machine and hasn't been tested here.

Explicitly *not* a gap: §3's aside about benchmarking NVIDIA Parakeet-TDT
is framed there as "a v2 investigation, not a v1 dependency" -- it was
never part of the required build order in the first place.

## Building

Requires the Rust toolchain, `cmake`, and a `libclang` bindgen can find via
`LIBCLANG_PATH` (`asr`, `vad`, and `cleanup` all compile a C/C++ inference
engine from source). See `crates/asr/README.md` for the specifics.
`tray-app` additionally pulls in `eframe`/`egui` (windowing + rendering)
and `tray-icon` -- no extra native toolchain beyond what's already
required, just a longer first build.

```sh
cargo build --workspace
cargo test --workspace    # 83 tests, all pure-logic; hardware/model paths are compile-verified only
cargo clippy --workspace --all-targets -- -D warnings
```

Model weights are not checked into the repo (see `.gitignore`) -- see
"Running" above, or each crate's own README, for fetch instructions.
