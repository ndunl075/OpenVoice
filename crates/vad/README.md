# vad

Silero VAD endpointing via [`ort`](https://ort.pyke.io/) (ONNX Runtime).
See `dictation-architecture.md` §2.2.

## Fetching a model

Not checked into the repo (binary, not source -- see the root
`.gitignore`). Grab the ONNX export from the official Silero VAD repo:

```sh
mkdir -p models
curl -L -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

`SileroVad::load` expects this file's v4 graph shape: inputs `input`
`[1, 512]` f32, `sr` `[1]` i64, `h`/`c` `[2, 1, 64]` f32; outputs `output`
`[1, 1]` f32, `hn`/`cn` `[2, 1, 64]` f32. If upstream ships a newer graph
with a different I/O contract, `process_frame` will fail at the
`try_extract_tensor`/named-input calls with a clear `ort` error rather than
silently misreading tensors.

## Native build requirements

None beyond what `cargo build` needs on its own -- `ort`'s default
features (`download-binaries`, `copy-dylibs`) fetch a prebuilt ONNX
Runtime binary at build time and place the dylib next to the compiled
output. No system ONNX Runtime install or cmake step required (unlike
`asr`, which does compile whisper.cpp from source).
