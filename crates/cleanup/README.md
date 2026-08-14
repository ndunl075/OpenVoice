# cleanup

Deadlined disfluency-cleanup pass via [`llama-cpp-2`](https://github.com/utilityai/llama-cpp-rs)
(llama.cpp). See `dictation-architecture.md` §2.4.

## Fetching a model

Not checked into the repo (binary, not source -- see the root
`.gitignore`). Grab a quantized GGUF export of Qwen2.5-0.5B-Instruct:

```sh
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Any instruct-tuned model that speaks ChatML should work with
[`build_prompt`](src/prompt.rs) -- Qwen2.5-0.5B-Instruct is what the doc
calls out specifically, chosen for being small enough to plausibly beat
the 120ms deadline on CPU.

## The deadline is load-bearing, not a nice-to-have

`CleanupModel::clean` races [`CLEANUP_DEADLINE`] (120ms). If generation
hasn't finished by then, the call returns `None` and the abandoned
generation keeps running on its own thread to completion -- nothing
waits for it, nothing reads its result. The caller (the daemon) is
expected to insert the raw ASR text in that case and never revisit the
decision once made: §2.4 is explicit that inserting raw text and then
patching it after the fact is worse than slightly rougher output, because
it means editing text the user may already be interacting with.

## Native build requirements

Same as `asr`: `cmake` on `PATH`, a C/C++ toolchain, and `LIBCLANG_PATH`
pointed at a `libclang` bindgen can use. See `crates/asr/README.md` for
details -- both crates compile a C++ inference engine via the same kind
of `cmake`+`bindgen` build.
