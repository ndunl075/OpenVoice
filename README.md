# OpenVoice

On-device voice dictation: you hold a hotkey, talk, let go, and the text
lands at your cursor — and no audio ever leaves the machine. Full design
rationale lives in [`dictation-architecture.md`](dictation-architecture.md) —
this README tracks what's actually built.

**On the "< 200 ms" target:** the architecture doc sets that as the goal,
and this README used to claim it was met. It isn't, and nobody had
measured it — see [Latency: the real numbers](#latency-the-real-numbers)
below for what it actually does today (roughly **3.5 seconds** from key
release to text on the hardware this was built on). That is *slower*
than the cloud tools this was meant to beat — being honest about it is
more useful than a number that flatters the design. What this project
does deliver, verifiably, is everything in "Why use this anyway" below:
no audio ever leaves the machine, it works with no network at all, and
it captures the half-second *before* you press the key.

(The GitHub repo is renamed to `OpenVoice` -- old `wispr-flow-clone`
clone URLs still redirect. The local working-copy folder on disk here
is still named `wispr flow clone`; ask if you want that renamed too,
since that one can disrupt an open editor/terminal mid-session in a way
the GitHub-side rename doesn't.)

## Status

The architecture doc's full build order is implemented: **v0 → v1 → v2**,
each merged as its own reviewed PR with a green CI run.

- [x] **v0 — demo pipeline:** ring buffer, push-to-talk hotkey, batch Whisper
      on release, clipboard-paste injection.
- [x] **v1 — the real win:** Silero VAD endpointing, streaming windowed decode.
- [x] **v2 — quality:** deadlined cleanup LLM pass (built, but **off by
      default** — it can't meet §2.4's 120ms budget here, see below), user dictionary, pre-roll
      capture, hands-free mode.

That's every item §4's build order calls for, plus §3's "Tray + minimal
overlay" and a settings window.

**Feature-complete is not the same as target-met.** The build order is
done; the doc's headline **< 200 ms** latency target is not (see
[Latency: the real numbers](#latency-the-real-numbers)). Those are
separate claims and this README used to blur them. "Known gaps" below
covers what's in the doc but outside §4's checklist — of which GPU
backends is the one that actually blocks the latency target.

## Running

```sh
# 1. Fetch models (see crates/asr, crates/vad, crates/cleanup READMEs)
mkdir -p models
curl -L -o models/ggml-distil-small.en.bin \
  https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin
curl -L -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx
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

Hold Left Ctrl + Left Shift together to dictate, release either to insert
the transcribed text at the cursor -- or tap AltGr to switch to hands-free
mode (see below). It's a two-key chord rather than one key on purpose: a
single common modifier is easy to trip by accident (bumping Left Ctrl
while typing normally); a chord basically never happens unintentionally.
As of v1, transcription streams continuously while the chord is held
(rolling windows, §2.3 -- 1.2s with 0.2s overlap; see
`crates/asr/src/window.rs` for why that specific overlap fraction, not
just the sizes, is what keeps streaming from falling behind real-time)
instead of waiting for release -- only the trailing ~1s partial window is
left to decode at that point. The ASR model path
defaults to `models/ggml-distil-small.en.bin`; override it with a CLI arg
(`cargo run -p daemon --release -- path/to/model.bin`) or the
`DICTATION_MODEL_PATH` env var. The VAD model path defaults to
`models/silero_vad.onnx`; override with `DICTATION_VAD_MODEL_PATH`. The
cleanup model (§2.4) is **off by default and not even loaded** -- see
"The cleanup pass is off by default" below. Set
`DICTATION_ENABLE_CLEANUP=1` to load it (path:
`models/qwen2.5-0.5b-instruct-q4_k_m.gguf`, or
`DICTATION_CLEANUP_MODEL_PATH`); if it's enabled but missing, the daemon
logs that and runs without it rather than refusing to start.

## Why use this anyway

It is currently **slower** than the cloud dictation tools it was modelled
on, and its accuracy is that of `distil-small.en` — a small model — so it
will mishear more than a cloud service running a large one. Those are the
honest trade-offs. What it gives you in exchange is structural, not
incremental:

- **No audio ever leaves your machine.** Not "we don't store it" — there
  is no network code in the audio path at all, and
  [`crates/ring-buffer`](crates/ring-buffer) has no network dependency in
  its `Cargo.toml` to make that checkable rather than promised.
- **Works with no internet.** Not degraded-offline; there is nothing to
  degrade. On a plane, on a locked-down network, on an air-gapped box, it
  behaves identically.
- **Nothing is buffered to disk.** The rolling ~30s audio buffer is
  memory-only and continuously overwritten.
- **Pre-roll: it hears the half-second *before* you press the key.** A
  cloud tool structurally cannot do this — it isn't listening yet. If you
  start talking a beat early, you don't lose your first word.
- **No account, no subscription, no telemetry.** MIT licensed; fork it,
  change the hotkey, swap the model.

If you need the lowest possible latency and the best possible accuracy,
today, a cloud tool will serve you better and this README isn't going to
pretend otherwise. If you don't want a always-on microphone streaming to
someone else's servers, that's the trade this exists to make.

## The cleanup pass is off by default

A deliberate departure from §2.4, on measurement rather than taste.

§2.4 specifies a **hard 120ms deadline**: use the cleaned text if the
small LLM returns in time, otherwise insert raw. Measured on this
hardware ([`crates/cleanup/tests/abandoned_work_cost.rs`](crates/cleanup/tests/abandoned_work_cost.rs)),
a full generation takes **516-677ms**. It therefore *never once* beat
the deadline — meaning it never changed a single character of output,
while burning a burst of LLM inference on every utterance. Worse, until
recently that abandoned generation kept running to completion after the
deadline, so the cost was paid in full for a result that was discarded
by construction.

Memory saved by not loading it, measured rather than assumed: **~100MB**
of working set (771MB → 668MB, everything else identical). Worth stating
precisely, because the obvious guess is wrong — the model *file* is
460MB, but llama.cpp mmaps it, so resident use is far below that. The
CPU and heat saving is the bigger prize.

Two fixes: timed-out work is now genuinely cancelled (see
`cleanup::CancelToken`), and the model isn't loaded at all unless asked
for. §2.4's *quality* argument is sound — raw ASR output really is
disfluent — it's the 120ms budget that this machine can't meet. On
faster hardware, or with a raised deadline (accepting the latency), it's
worth turning back on.

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
utterances. Tap AltGr again to stop. Push-to-talk (Left Ctrl + Left
Shift) and hands-free are mutually exclusive: whichever session is
active, the other mode's key is ignored until it ends.

**Recording indicator:** `tray-app` (see [`crates/tray-app`](crates/tray-app))
gives you a system tray icon -- an original mic glyph, not a copy of any
product's logo, whose background color reflects pipeline state at a
glance -- plus a small floating pill near the bottom of the screen that
appears while the mic is live or the pipeline is working, and briefly
shows the result before fading. Push-to-talk-held and hands-free-armed
both read as "Listening…" (color still tells them apart -- coral for a
held key, lavender for hands-free) with an animated vertical-bar level
meter, the same genre convention most voice-input UIs use for "mic is
live right now"; "Cleaning up…" covers the deadlined cleanup pass. The
architecture doc calls a visible indicator out as necessary before an
always-on mic is trustworthy to ship publicly (see "Honest risks" in
`dictation-architecture.md`); this is that. The console binary (`daemon`)
still exists and still just prints to stdout, for headless/debugging use
-- `tray-app` is the one with an actual UI.

## Performance: why decoding used to be so slow

Real talk: for a while, actual transcription was badly slow -- multiple
*seconds* per short window, not the sub-200ms this whole project is
about. Three separate real, benchmarked (not guessed) issues, found and
fixed in sequence as each one surfaced the next:

1. **`audio_ctx` defaulting to a full 30s encoder context** regardless of
   actual audio length. Every decode call was paying for 30 seconds of
   encoder compute even on a 1-3 second clip. Scoping it to the real
   audio length ([`crates/asr/src/audio_ctx.rs`](crates/asr/src/audio_ctx.rs))
   measured an **~11.7x speedup** (a 3s clip: ~62s to decode → ~5s), for
   zero accuracy cost. Also switched the default model to
   `distil-small.en` (fewer decoder layers) and, on Windows/MSVC, fixed
   `whisper-rs-sys`'s CMake build silently landing on scalar-only code
   with every SIMD flag off (see `.cargo/config.toml`).
2. **Thread count clamped to 4** without ever checking whether that was
   actually the right number on real hardware -- it wasn't. See
   `crates/asr/README.md`'s "thread count wasn't actually tuned either"
   for the numbers (4 threads: slower than real-time; 8: ~0.8x
   real-time; 16, all logical cores: 10-25x slower, badly).
3. **Overlap fraction too large for the new decode speed to sustain**:
   once decode got fast enough to matter, a large window-overlap
   fraction meant the pipeline would still silently fall behind over a
   long utterance, defeating the whole point of streaming ahead of the
   user finishing talking. See `crates/asr/src/window.rs`'s
   `default_16k` doc comment for the actual inequality this needs to
   satisfy.
4. **`audio_ctx` values that weren't block-aligned fell off ggml's SIMD
   kernels.** Fixing #1 introduced this one: scoping `audio_ctx` to the
   clip length produces small values for short clips, and small
   *non-multiple-of-16* values turn out to be catastrophic. Measured on
   identical 0.25s audio, varying only this parameter:
   `audio_ctx=13` → **5.2s**, `audio_ctx=16` → **327ms**. The fast
   values are exactly the multiples of 16; everything else costs ~10x.
   This hit the worst possible place — the trailing window at key
   release is short by construction, so it landed in the bad range
   constantly. Rounding up to a 16-frame block cut a 0.25s tail from
   **4.3s to 302ms**. See
   [`crates/asr/src/audio_ctx.rs`](crates/asr/src/audio_ctx.rs)'s
   `CONTEXT_BLOCK` and `crates/asr/tests/small_audio_ctx_probe.rs`.

If it ever feels slow again: benchmark it
(`crates/asr/README.md`'s "Benchmarking" section has the exact commands,
including a real-time-factor sweep across window sizes and thread
counts) before changing anything. Every one of the four regressions
above shipped for a while because nobody had actually timed a real
decode call until asked to — and #4 was found only because the *first*
guess about its cause (the temperature-fallback retry ladder) was
tested and turned out to be wrong.

## Latency: the real numbers

The architecture doc targets **< 200 ms** from end-of-speech to text at
the cursor. That target is **not currently met.** Measured with
[`crates/asr/tests/commit_latency.rs`](crates/asr/tests/commit_latency.rs)
(the numbers below are decode + the doc's own ~15ms injection estimate;
add up to 120ms more when the cleanup pass runs and uses its full
deadline):

| Trailing tail at release | End-of-speech → cursor |
|---|---|
| 0.10 s | ~40 ms |
| 0.25 s | ~3.4 s |
| 0.50 s | ~3.7 s |
| 1.00 s (worst case) | ~3.7 s |

**These numbers got worse on purpose.** An earlier version of this table
read ~300ms–1.2s. That was measured while `audio_ctx` was scoped tightly
to clip length — which was fast, and produced *repetition loops* in real
use (a two-second utterance coming out as the same phrase ~30 times).
Fixing that required a much larger encoder context
(`asr::audio_ctx`'s `MIN_CONTEXT_FRAMES`), and a larger context costs
roughly linear time. Correct output at 3.7s beats garbage at 300ms, so
this is the right trade — but it is a real, large regression and
pretending otherwise would make the benchmark a lie.

Note the shape: cost is now dominated by the fixed context floor, not by
how much audio you actually spoke, which is why 0.25s and 1.0s tails
cost nearly the same.

**Why it can't currently hit 200 ms:** the tail is whatever audio
arrived since the last full streaming window, so it's bounded by the
window stride (1.0s). Getting a 200ms *total* would need the tail under
roughly 0.2s of audio, i.e. a ~0.2s stride — but streaming only keeps up
if each window decodes faster than the next one arrives (`0.8 × window <
stride`, see `window.rs`), which with a 0.2s stride would force windows
so short that accuracy collapses. On this CPU, with this model, the two
constraints are mutually exclusive. **GPU acceleration is the real path
to the doc's target** — see "Known gaps" below; it's the difference
between shaving milliseconds and changing the constraint.

Two honest caveats on the numbers above:

- **The benchmark drives a synthetic sine, not real speech.** It forces
  real encoder/decoder work, but token counts — and so decode time —
  differ on real content. Treat these as the right order of magnitude,
  not gospel.
- **There's a worst case worse than the table.** whisper.cpp's
  temperature-fallback retry ladder re-decodes when its own confidence
  checks flag a bad result. That's deliberately enabled here (it's what
  fixed a real "types random words" bug — see
  `AsrConfig::temperature_inc`), and it's a genuine tradeoff: on audio
  the model finds ambiguous, a tail decode can spike to **~5s**. The
  synthetic sine triggers this on *every* run because it isn't speech at
  all; real speech should trigger it rarely. Quantifying "rarely"
  honestly needs a corpus of real recorded audio, which this repo
  doesn't have yet — flagging it rather than guessing a number.

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

- **GPU backends -- and this is the one that matters.** §2.3 mentions
  Metal/CUDA backends for whisper.cpp; `asr`'s `Cargo.toml` doesn't
  enable whisper-rs's `cuda`/`metal` feature flags, so this build is
  CPU-only (whisper.cpp's own AVX2 auto-detection still applies --
  that's the "CPU/AVX2 fallback" leg of §2.3's backend list, just not
  the GPU-accelerated ones).

  This is no longer a nice-to-have: per [Latency: the real
  numbers](#latency-the-real-numbers), CPU decode speed is exactly what
  makes the doc's < 200 ms target unreachable, because it forces a
  window stride far larger than the latency budget allows. Every
  remaining CPU-side optimization is shaving milliseconds off the wrong
  constraint. The machine this was built on has no NVIDIA GPU (so no
  CUDA) and isn't Apple (so no Metal) — it has an Intel Arc 140T
  integrated GPU, which makes **Vulkan** the realistic backend to try.
  The Cargo features are now wired up and ready — `asr`, `daemon`, and
  `tray-app` each expose `vulkan` / `cuda` / `metal`, all **off by
  default** (a default-on GPU feature would turn `cargo build` into a
  confusing native-build failure on every machine without the vendor
  SDK, CI included):

  ```sh
  cargo build -p tray-app --release --features vulkan
  ```

  What's *not* done is actually building and benchmarking it. That needs
  the **Vulkan SDK** installed at build time — this machine has the
  Vulkan runtime loader (`vulkan-1.dll`) and a capable GPU, but not the
  SDK, so the build above hasn't been run or verified here. Until
  someone does that and re-runs `commit_latency`, treat "Vulkan will fix
  the latency target" as the reasonable hypothesis it is, not a measured
  result.

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
cargo test --workspace    # 104 tests, all pure-logic; hardware/model paths are compile-verified only
cargo clippy --workspace --all-targets -- -D warnings
```

Model weights are not checked into the repo (see `.gitignore`) -- see
"Running" above, or each crate's own README, for fetch instructions.
