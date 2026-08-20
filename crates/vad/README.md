# vad

Silero VAD endpointing via [`ort`](https://ort.pyke.io/) (ONNX Runtime).
See `dictation-architecture.md` §2.2.

## Fetching a model

Not checked into the repo (binary, not source -- see the root
`.gitignore`). Grab the ONNX export from the official Silero VAD repo:

```sh
mkdir -p models
curl -L -o models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx
```

Note the **pinned tag** rather than `master`. This crate is written
against Silero **v5**'s graph, and a silent upstream bump is exactly the
kind of thing that broke it before (see below).

## The v4 -> v5 contract, and how it failed silently

`SileroVad` expects v5's shape: inputs `input` `[1, 576]` f32, `state`
`[2, 1, 128]` f32, `sr` scalar i64; outputs `output` `[1, 1]` f32,
`stateN` `[2, 1, 128]` f32.

It was originally written against **v4** (`h`/`c` `[2, 1, 64]` in,
`hn`/`cn` out), and the README used to claim a mismatch would "fail with
a clear `ort` error rather than silently misreading tensors." Half of
that turned out to be true, and the other half caused a bug that hid for
a long time:

1. **The name change did error** -- `Invalid input name: h`, on every
   frame. But `daemon`'s `feed_vad` only logs frame failures to stderr,
   and the GUI build has no console, so nobody ever saw it. The VAD
   simply never ran: no speech was ever detected, streaming decode never
   engaged, and every utterance fell back to a single large decode at
   hotkey release.
2. **The context change did *not* error.** v5 expects 64 samples of the
   *previous* chunk prepended to each frame (64 + 512 = 576), and its
   `input` is declared with a fully dynamic shape `[-1, -1]` -- so
   feeding a bare 512-sample chunk is accepted and just returns a
   meaningless number. Measured on the bare chunk: silence `0.0006`,
   clear speech `0.0006`, loud noise `0.0011`. With the context
   prepended, the same clip peaks at `1.000` and confirms speech at
   frame 7.

That second one is the lesson worth keeping: a dynamic input shape means
the model will happily accept the wrong thing. `tests/real_speech_probe.rs`
exists so this is checkable against real speech in one command instead of
being inferred from the code.

## Native build requirements

None beyond what `cargo build` needs on its own -- `ort`'s default
features (`download-binaries`, `copy-dylibs`) fetch a prebuilt ONNX
Runtime binary at build time and place the dylib next to the compiled
output. No system ONNX Runtime install or cmake step required (unlike
`asr`, which does compile whisper.cpp from source).
