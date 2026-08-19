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

Three `#[ignore]`d integration tests exist specifically so "is this slow?"
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
