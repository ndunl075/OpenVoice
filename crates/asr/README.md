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

Two `#[ignore]`d integration tests exist specifically so "is this slow?"
never has to be answered by guessing again:

```sh
# Compare decode wall-clock time between two models
ASR_BENCH_MODEL_A=models/ggml-small.en-q5_1.bin \
ASR_BENCH_MODEL_B=models/ggml-distil-small.en.bin \
cargo test --release -p asr --test model_benchmark -- --ignored --nocapture

# Confirm audio_ctx scoping is actually helping on your hardware
ASR_BENCH_MODEL_A=models/ggml-small.en-q5_1.bin \
cargo test --release -p asr --test audio_ctx_experiment -- --ignored --nocapture
```

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
