# Local Dictation Engine — Architecture

**Thesis:** on-device voice dictation where text appears effectively instantly, and no audio ever leaves the machine.

**Target:** < 200 ms from end-of-speech to text at cursor. Cloud tools land in the 500 ms – 1.5 s range because they pay a network round trip on every utterance. That gap is the entire product.

---

## 1. Where the latency actually goes

Most dictation tools are slow for reasons that have nothing to do with model inference. The naive pipeline:

| Stage | Naive cost | Our cost | How |
|---|---|---|---|
| Mic device init on hotkey | 100–300 ms | 0 ms | Always-on ring buffer |
| Wait for user to release key | — | — | (user-controlled) |
| Detect end of speech | 500–800 ms fixed timeout | 30–90 ms | Silero VAD endpointing |
| Upload audio | 50–400 ms | 0 ms | Local |
| Model cold start / load | 200 ms – 2 s | 0 ms | Resident daemon, model pinned in memory |
| Transcribe full utterance | 300 ms – 1 s | 40–120 ms | Streamed during speech; only tail remains |
| Download result | 30–200 ms | 0 ms | Local |
| Insert text | 50–300 ms | ~15 ms | Clipboard swap + synthetic paste |

The headline insight: **transcription should be nearly finished before the user stops talking.** Batch-after-release is the single biggest self-inflicted wound in most implementations.

---

## 2. Component architecture

Single native daemon. No Electron, no Python runtime, no IPC across process boundaries in the hot path.

```
┌─────────────────────────────────────────────────────┐
│  Daemon (Rust, tray app, always resident)           │
│                                                     │
│  ┌───────────┐   ┌─────────┐   ┌──────────────┐     │
│  │ Audio ring│──▶│ VAD     │──▶│ Streaming    │     │
│  │ buffer    │   │ (Silero)│   │ ASR worker   │     │
│  │ 16k mono  │   └─────────┘   └──────┬───────┘     │
│  │ 30s circ. │                        │             │
│  └─────▲─────┘                        ▼             │
│        │                       ┌──────────────┐     │
│  ┌─────┴─────┐                 │ Cleanup pass │     │
│  │ Global    │                 │ (small LLM,  │     │
│  │ hotkey    │                 │  deadlined)  │     │
│  └───────────┘                 └──────┬───────┘     │
│                                       ▼             │
│                                ┌──────────────┐     │
│                                │ Text injector│     │
│                                └──────────────┘     │
└─────────────────────────────────────────────────────┘
```

### 2.1 Always-on audio ring buffer
- `cpal`, 16 kHz mono f32, 30-second circular buffer, opened once at daemon start.
- Kills device-init latency entirely.
- **Bonus feature this unlocks:** pre-roll. Capture the ~500 ms *before* the hotkey press. Solves the universal "I started talking a beat too early and lost my first word" problem. Cloud tools structurally cannot do this. Good demo moment.
- Privacy note: buffer is in-memory only, never written to disk, overwritten continuously. Say this loudly and make it verifiable in the repo, since an always-on mic is exactly what a privacy-positioned product cannot be sloppy about.

### 2.2 VAD / endpointing
- Silero VAD via ONNX Runtime. ~1 MB, sub-millisecond per 30 ms frame.
- Two jobs: (a) trim silence so the encoder never wastes compute on dead air, (b) detect end-of-speech in ~2–3 frames instead of a fixed timeout.
- Supports both interaction modes: push-to-talk (release = commit) and hands-free (VAD silence = commit).

### 2.3 Streaming ASR
- `whisper.cpp` via `whisper-rs`. Model: `distil-small.en` or `small.en`, Q5_K quantized.
- Rolling 3 s windows with 0.5 s overlap, decoded *while the user speaks*. At release, only the final partial window is outstanding.
- Decode settings that matter for speed:
  - `beam_size = 1` (greedy) — largest single win, modest WER cost
  - no temperature fallback
  - `condition_on_previous_text = false` — prevents hallucination loops on short utterances
  - `suppress_blank`, `no_timestamps` when timestamps aren't needed
- Backends: Metal (Apple Silicon), CUDA (your Windows box), CPU/AVX2 fallback.
- Custom vocab: bias decoding with an `initial_prompt` seeded from a user dictionary (names, jargon, product names). Cheap, high perceived accuracy gain.

### 2.4 Cleanup pass (optional, deadlined)
Raw ASR output is disfluent: "um", false starts, no punctuation. Cloud competitors run an LLM to fix this, and it's a real part of why their output feels good.

- Qwen2.5-0.5B-Instruct Q4 via `llama.cpp`, resident.
- **Hard deadline: 120 ms.** If it returns in time, insert cleaned text. If not, insert raw and drop the cleanup.
- Do *not* insert raw text and then patch it afterward. Editing text the user may already be interacting with is worse than slightly rougher output. Pick one string, insert once.

### 2.5 Text injection
- Primary: save clipboard → set clipboard → synthetic Cmd/Ctrl+V → restore clipboard after ~50 ms. Fastest and most universally compatible path.
- Fallback: per-character synthetic keystrokes for apps that block programmatic paste (some terminals, secure/password fields).
- Never inject into a field flagged as secure input.

---

## 3. Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Rust | Predictable latency, no GC pauses in hot path, consistent with your other tooling |
| Audio | `cpal` | Cross-platform, low-level enough |
| VAD | Silero via `ort` | Tiny, fast, well-proven |
| ASR | `whisper-rs` (whisper.cpp) | Easiest quality embeddable engine with GPU backends |
| Cleanup LLM | `llama.cpp` | Same |
| Hotkeys | `rdev` / platform APIs | Global capture |
| UI | Tray + minimal overlay | Electron would undo the entire premise |

Alternative ASR worth benchmarking before committing: NVIDIA Parakeet-TDT. Meaningfully faster real-time factor than Whisper, but it's NeMo and harder to embed cleanly, so it's a v2 investigation, not a v1 dependency.

---

## 4. Build order

**v0 (~2 days) — the demo.** Ring buffer + push-to-talk hotkey + batch Whisper on release + clipboard injection. No VAD, no streaming, no cleanup. This alone already beats cloud tools on latency because you deleted the network. Ship the clip here.

**v1 (~3 days) — the real win.** Add Silero VAD endpointing and streaming windowed decode. This is where you cross under 200 ms.

**v2 — quality.** Cleanup LLM with deadline, user dictionary, pre-roll capture, hands-free mode.

Do not build v2 before shipping v0 publicly. The series dies if episode one takes three weeks.

---

## 5. Honest risks

**Latency has a perceptual floor.** Below roughly 150–200 ms, users cannot distinguish improvements. Optimizing 90 ms down to 60 ms is engineering vanity. The marketable difference is "instant" versus "waiting," not one benchmark number versus another. Stop optimizing once it feels instant and spend the remaining time on output quality.

**Latency is probably not the incumbent's actual moat.** Wispr Flow's stickiness comes more from formatting quality, context-awareness of the active app, and tone adaptation than from raw speed. If you win on milliseconds and lose on output quality, you have built a worse product with a better benchmark. Budget real time for the cleanup pass.

**Accuracy/speed tradeoff is real.** `distil-small.en` versus `large-v3` is a genuine WER difference, and it widens on accented speech, technical jargon, and proper nouns. If your demo is you speaking clearly in a quiet room, it will look better than it is. Test on your actual worst-case audio before claiming parity.

**Always-on mic is a trust liability.** It is the correct engineering decision and the riskiest optics decision. Make the buffer behavior explicit in the README, keep it memory-only, and add a visible recording indicator.

**Demo problem:** "faster" is invisible in a screen recording unless you make it visible. Film a side-by-side against a cloud tool with an on-screen timer, or the entire premise won't land with viewers.
