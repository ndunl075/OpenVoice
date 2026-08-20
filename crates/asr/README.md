# asr

Batch/streaming transcription via [`whisper-rs`](https://github.com/tazz4843/whisper-rs)
(whisper.cpp). See `dictation-architecture.md` §2.3.

## Performance: `audio_ctx` matters more than which model you pick

See [`src/audio_ctx.rs`](src/audio_ctx.rs) for the full writeup, but the
short version: whisper.cpp's `audio_ctx` decode parameter defaults to `0`,
which means "encode a full 30-second context" **no matter how much audio
you actually pass in**. Every `transcribe()` call was silently paying for
30 seconds of encoder compute even on a 1-second clip. Measured on real
hardware: a 3-second clip went from ~62s to decode down to ~5s once
`audio_ctx` was scoped to the clip's actual length -- an ~11.7x speedup,
for free, no accuracy tradeoff. This is wired in automatically
(`Transcriber::transcribe` computes it from the audio you pass); there's
nothing to configure.

If dictation ever feels slow again, benchmark before guessing why -- see
"Benchmarking" below. This crate shipped with that ~12x regression
sitting in it for a while because nobody had actually timed a real decode
call until asked to.

## Performance: thread count wasn't actually tuned either

[`default_thread_count`](src/lib.rs) used to just mirror whisper.cpp's own
upstream default (`min(4, hardware_concurrency)`) -- a reasonable-sounding
number nobody had actually benchmarked *on this machine*. Turns out it
mattered: on a 16-logical-core machine, `real_time_factor.rs` (see
"Benchmarking") measured `distil-small.en` decoding at ~1.1-1.3x
real-time at 4 threads -- i.e. genuinely *slower than the audio itself*,
which means the whole streaming design (§2.3: "transcription should be
nearly finished before the user stops talking") can't keep its promise no
matter how the windows are scheduled. At 8 threads: ~0.8x, comfortably
real-time. At 16 (all logical cores): catastrophically *worse* -- 10-25x
real-time, almost certainly thread-pool/scheduling contention once you
cross from physical into hyperthreaded cores. More threads is not
monotonically better; `default_thread_count` now targets roughly the
physical core count (`available_parallelism / 2`, clamped to `[1, 8]`)
instead of either extreme.

## Performance: small `audio_ctx` values fall off ggml's SIMD kernels

Scoping `audio_ctx` to the clip length (above) introduced a second,
subtler problem: short clips produce *small* `audio_ctx` values, and
small values that aren't multiples of 16 are catastrophically slow.
Measured on identical 0.25s audio, varying only this one parameter
(`tests/small_audio_ctx_probe.rs`):

| `audio_ctx` | time | | `audio_ctx` | time |
|---|---|---|---|---|
| 8 | 3.85s | | **16** | **327ms** |
| 12 | 4.93s | | **24** | **432ms** |
| 13 | 5.21s | | **32** | **578ms** |
| 14 | 5.06s | | | |
| 18 | 5.27s | | | |
| 20 | 4.37s | | | |
| 22 | 4.44s | | | |

The fast values are exactly the multiples of 16; everything else costs
roughly 10x on the same audio. A hard fast/slow split on block-size
boundaries (rather than a smooth curve) is the signature of ggml
dropping off its SIMD-blocked matmul kernels onto a scalar path —
consistent with this build having to force the AVX2/FMA flags on
explicitly in the first place (see the repo's `.cargo/config.toml`).

`audio_ctx_for` now rounds up to a 16-frame block. This mattered most in
the worst possible place: the trailing partial window at hotkey release
is short by construction, so it was landing in the pathological range
constantly. Fixing it cut a 0.25s tail from **4.3s to 302ms**.

Worth noting how this was found, since the method generalizes: the first
hypothesis was whisper.cpp's temperature-fallback retry ladder, which
was *wrong* — sweeping `temperature_inc` (including 0.0, which disables
the ladder entirely) reproduced the spike unchanged. Only then did
holding the audio fixed and sweeping `audio_ctx` alone isolate the real
cause.

## Deliberate deviation from the architecture doc: temperature fallback

`dictation-architecture.md` §2.3 lists "no temperature fallback" among
its speed settings. This crate **does not follow that**, and the
deviation is intentional — see `AsrConfig::temperature_inc`.

Disabling the fallback entirely (`temperature_inc = 0.0`) is genuinely
fastest, but it shipped real, user-visible garbage output: with nothing
to catch a bad greedy decode, short context-free windows (which is every
window here) would confidently emit nonsense. The fallback is the
mechanism that re-decodes when whisper's own entropy/logprob checks flag
a result as bad.

The cost is worst-case latency, and it isn't small: each rung of the
ladder is another full decode. `DEFAULT_TEMPERATURE_INC` is 0.4 rather
than whisper.cpp's own 0.2 to bound the ladder to 3 rungs instead of 6.
Even so, a tail decode on audio the model finds ambiguous can spike to
~5s (see the root README's latency section). This is a real, unresolved
tradeoff, not a solved problem — tuning `entropy_thold`/`logprob_thold`
to make the fallback trigger less eagerly is the obvious next lever, but
doing that honestly needs a corpus of real recorded speech to validate
against, which this repo doesn't have yet.

## Fetching a model

Model weights aren't checked into the repo (see the root `.gitignore`) --
they're a multi-hundred-MB binary download, not source.

```sh
mkdir -p models
curl -L -o models/ggml-distil-small.en.bin \
  https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin
```

This is the default (`daemon::model_path()`). `distil-small.en` has far
fewer decoder layers than `small.en` and decodes measurably faster on top
of the `audio_ctx` fix above -- it's also what the architecture doc calls
out as the speed-oriented pick. Note the file hosted here is fp16, not
quantized (no pre-quantized ggml build is published for it); it's still
faster than `small.en` despite being a larger download, because layer
count -- not quantization -- is what drives decode time.

`small.en`, the safer pick for accuracy, still works the same way:

```sh
curl -L -o models/ggml-small.en-q5_1.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin
```

Either works with `AsrConfig::new(path)` -- point `model_path` at
whichever `.bin` you fetched (or set `DICTATION_MODEL_PATH`).

## Benchmarking

Five `#[ignore]`d integration tests exist specifically so "is this slow?"
never has to be answered by guessing again:

```sh
# Compare decode wall-clock time between two models
ASR_BENCH_MODEL_A=models/ggml-small.en-q5_1.bin \
ASR_BENCH_MODEL_B=models/ggml-distil-small.en.bin \
cargo test --release -p asr --test model_benchmark -- --ignored --nocapture

# Confirm audio_ctx scoping is actually helping on your hardware
ASR_BENCH_MODEL_A=models/ggml-small.en-q5_1.bin \
cargo test --release -p asr --test audio_ctx_experiment -- --ignored --nocapture

# Real-time factor per WindowPolicy-sized window, swept across thread
# counts -- the tool that found the thread-count and overlap-fraction
# issues above. Use this one first if streaming feels behind real-time.
ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin \
cargo test --release -p asr --test real_time_factor -- --ignored --nocapture

# End-of-speech -> text-at-cursor latency: the number the architecture
# doc's "< 200 ms" target is actually about. Reports rather than
# asserts; also sweeps temperature_inc so the retry ladder's cost is
# visible. Use this one if dictation feels laggy *on release*
# specifically (as opposed to falling behind while you talk).
ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin \
cargo test --release -p asr --test commit_latency -- --ignored --nocapture

# Holds audio fixed and sweeps audio_ctx alone -- how the SIMD
# block-alignment cliff above was isolated. Reach for this shape of
# experiment when a result makes no sense (e.g. shorter input decoding
# *slower*), to separate one variable from everything else.
ASR_BENCH_MODEL_A=models/ggml-distil-small.en.bin \
cargo test --release -p asr --test small_audio_ctx_probe -- --ignored --nocapture
```

The synthetic swept-sine signal these use is a real caveat, not a
formality: it exercises real encoder+decoder compute, but token count
(and so decode time) can genuinely differ for real speech content. Numbers
here are a strong signal, not a guarantee -- if streaming still feels off
after tuning against them, that's worth re-checking against an actual
recording.

Not run in CI (they need local model files and are meant for a human to
read the numbers), and use `--release` -- debug-profile whisper.cpp is
not representative of real performance.

## Native build requirements

`whisper-rs` compiles whisper.cpp (C++) via `cmake`. You need:
- `cmake` on `PATH`
- a C/C++ toolchain cmake can find a generator for (MSVC on Windows, or
  clang/gcc + Ninja/Make)
- `LIBCLANG_PATH` set to a directory containing `libclang.dll`/`.so` (used
  by `bindgen` to generate the FFI bindings) -- the LLVM-MinGW toolchain
  some setups use for cross-compiling does *not* ship this; a full LLVM
  release does.

On Windows/MSVC specifically, the repo's root `.cargo/config.toml` forces
`GGML_AVX2`/`GGML_FMA`/`GGML_F16C`/`GGML_SSE42` on for whisper.cpp's CMake
build. Without it, `GGML_NATIVE=ON`'s CPU auto-detection is a silent
no-op under MSVC (no `-march=native` equivalent), leaving every SIMD flag
off and ggml running scalar-only matmuls. See that file's comments for
the full explanation.
