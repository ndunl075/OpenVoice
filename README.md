# Local Dictation Engine

On-device voice dictation: text appears at the cursor in well under 200 ms
after you stop talking, and no audio ever leaves the machine. Full design
rationale lives in [`dictation-architecture.md`](dictation-architecture.md) —
this README tracks what's actually built.

## Status

Working through the architecture doc's build order: **v0 → v1 → v2**. See
the checklist below; boxes are checked as each piece lands on `main`.

- [x] **v0 — demo pipeline:** ring buffer, push-to-talk hotkey, batch Whisper
      on release, clipboard-paste injection.
- [ ] **v1 — the real win:** Silero VAD endpointing, streaming windowed decode.
- [ ] **v2 — quality:** deadlined cleanup LLM pass, user dictionary, pre-roll
      capture, hands-free mode.

## Running v0

```sh
# 1. Fetch a model (see crates/asr/README.md)
mkdir -p models
curl -L -o models/ggml-small.en-q5_1.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin

# 2. Run the daemon
cargo run -p daemon --release
```

Hold Right Ctrl to dictate, release to insert the transcribed text at the
cursor. The model path defaults to `models/ggml-small.en-q5_1.bin`; override
it with a CLI arg (`cargo run -p daemon --release -- path/to/model.bin`) or
the `DICTATION_MODEL_PATH` env var.

**Recording indicator, honestly scoped:** v0 prints recording state to the
console (it's the only UI that exists yet) -- there's no persistent
OS-level tray icon showing "mic is live" independent of that terminal
window. The architecture doc calls a visible indicator out as necessary
before this is trustworthy to ship publicly (see "Honest risks" in
`dictation-architecture.md`); a real tray icon is tracked as follow-up
work, not silently assumed to exist.

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
- The tray app shows a visible recording indicator whenever the mic stream
  is active, so "always-on" is never invisible.
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
| [`daemon`](crates/daemon) | §2 | Binary that wires all of the above together |

## Building

Requires the Rust toolchain and `cmake` (for the whisper.cpp / ONNX Runtime
native builds pulled in by `asr` and `vad`).

```sh
cargo build --workspace
cargo test --workspace
```

Model weights are not checked into the repo (see `.gitignore`) — fetch
instructions land alongside the ASR crate once it's implemented.
