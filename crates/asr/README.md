# asr

Batch/streaming transcription via [`whisper-rs`](https://github.com/tazz4843/whisper-rs)
(whisper.cpp). See `dictation-architecture.md` §2.3.

## Fetching a model

Model weights aren't checked into the repo (see the root `.gitignore`) --
they're a multi-hundred-MB binary download, not source. Grab a quantized
ggml model from the official whisper.cpp model repo:

```sh
mkdir -p models
curl -L -o models/ggml-small.en-q5_1.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin
```

`distil-small.en` (from
[distil-whisper](https://huggingface.co/distil-whisper)) is the faster
alternative called out in the architecture doc; `small.en` is the safer
default for accuracy. Either works with `AsrConfig::new(path)` -- point
`model_path` at whichever `.bin` you fetched.

## Native build requirements

`whisper-rs` compiles whisper.cpp (C++) via `cmake`. You need:
- `cmake` on `PATH`
- a C/C++ toolchain cmake can find a generator for (MSVC on Windows, or
  clang/gcc + Ninja/Make)
- `LIBCLANG_PATH` set to a directory containing `libclang.dll`/`.so` (used
  by `bindgen` to generate the FFI bindings) -- the LLVM-MinGW toolchain
  some setups use for cross-compiling does *not* ship this; a full LLVM
  release does.
