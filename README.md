# Offline Real-Time Voice Assistant

A fully offline, audio-in / audio-out conversational assistant. You speak, it transcribes,
a local language model answers, and it speaks the answer back — with no network calls at any
stage. The design target is low *perceived* latency: the user should hear the beginning of a
reply quickly, even when the full response takes longer to generate.

Built and measured on a Lenovo LOQ 15ARP9 (Ryzen 7 7435HS, RTX 4050 6 GB, 24 GB RAM),
Fedora 44.

## How it works

```
mic ─► Silero VAD ─► filler ack ─► faster-whisper ─► qwen2.5:3b (Ollama) ─► Piper ─► speaker
        (endpoint)   (<300 ms)       (STT, GPU)        (LLM, streamed)      (TTS, CPU)
                                                             │
                                          sentence-by-sentence hand-off
```

Two things keep it responsive:

- **Sentence streaming.** Tokens are streamed from the LLM and split on sentence
  boundaries. The first complete sentence is synthesized and played while the model is still
  generating the rest, so audio starts long before the full answer is ready.
- **Instant acknowledgement.** The moment the user stops speaking, a short pre-rendered
  filler ("Mhm.", "Let me think.") plays while STT and the LLM start, so there is never a
  silent gap.

STT runs on the GPU; TTS deliberately runs on the CPU so that Whisper and the 3B model can
share the 6 GB of VRAM without contention.

### Fallback behaviour

The assistant is built not to stall or show errors:

- A **watchdog** plays a stall line if the first token is late (1.5 s), and an honest
  holding reply that reopens the microphone if generation is still stuck (5 s).
- If **Ollama is unavailable** — even killed mid-conversation — the assistant acknowledges,
  apologizes briefly, and invites the user to continue. It degrades; it does not crash.
- **Barge-in:** if the user talks over the assistant, playback is flushed and the in-flight
  generation is cancelled so it starts listening again.

## Measured latency

Median per stage, relative to end of speech, warm (RTX 4050, model resident in VRAM):

| Stage | Median |
|---|---|
| Filler acknowledgement | ~3 ms |
| STT (faster-whisper small.en) | ~110 ms |
| LLM first token | ~335 ms |
| **Time to first audio (TTFA)** | **~690 ms** |
| Full reply spoken | ~850 ms |

GPU memory with STT + LLM + TTS all loaded: ~3.6 GB of 6 GB.

Latency is logged per turn to `latency_log.csv`; regenerate it by running the assistant or
`test_turn.py`.

## Setup

Requires Python 3.10, an NVIDIA GPU with CUDA, PortAudio, and Ollama.

```bash
# 1. system audio library (Fedora; on Debian/Ubuntu: sudo apt install libportaudio2)
sudo dnf install portaudio              # or: conda install -c conda-forge portaudio

# 2. python dependencies
pip install -r requirements.txt

# 2b. only if your torch/CUDA stack is CUDA 13: ctranslate2 (faster-whisper's backend)
#     links against CUDA 12, so install the CUDA 12 libs as wheels — run.sh points
#     LD_LIBRARY_PATH at them automatically
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# 3. language model (Ollama must be installed and running)
ollama pull qwen2.5:3b

# 4. Piper voice
python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

The `INPUT_DEVICE_MATCH` / `OUTPUT_DEVICE_MATCH` values in `config.py` select the audio
device by name substring; adjust them for your machine (`python -m sounddevice` lists the
available devices).

## Running

```bash
./run.sh                     # recommended launcher
# or:
python main.py               # start the assistant directly; Ctrl+C prints the latency summary
python audio_smoketest.py    # optional: confirm mic + speaker first
```

The pipeline opens the onboard audio codec directly through ALSA. On a PipeWire system that
device is normally held by the sound server, so `run.sh` switches the analog card's profile
off for the session (releasing it) and restores it on exit; other cards are untouched. If
card detection picks the wrong device, set it explicitly: `AUDIO_CARD=alsa_card.xxx ./run.sh`
(list cards with `pactl list cards short`). Running `python main.py` directly assumes the
analog device is already free.

Headphones are recommended: on open laptop speakers the assistant's own voice can leak into
the microphone and interfere with barge-in detection.

## Project layout

| File | Responsibility |
|---|---|
| `main.py` | Main loop, turn orchestration, sentence streaming, barge-in |
| `config.py` | All tunable parameters |
| `audio_io.py` | Device lookup, resampling, streaming playback |
| `vad.py` | Silero VAD endpointing |
| `stt.py` | faster-whisper speech-to-text |
| `llm.py` | qwen2.5:3b via Ollama, streamed |
| `tts.py` | Piper text-to-speech |
| `state.py` | Conversation states and fallback phrases |
| `metrics.py` | Per-turn latency logging |
| `test_turn.py` | Headless pipeline test (no microphone) |
| `audio_smoketest.py` | Microphone / speaker check |

## Design notes

- **VAD:** Silero rather than WebRTC/energy VAD — much less prone to false triggers on fan
  and keyboard noise, and cheap enough to run per-frame on the CPU.
- **STT model:** `small.en` is the balance point on this GPU; `base.en` drops technical
  words and `medium` is too slow for the latency budget.
- **LLM size:** a 3B model keeps time-to-first-token low; a 7B model does not fit the 2 s
  budget on a 6 GB laptop GPU.
- **Sample rate:** the onboard codec only accepts 44.1k/48k over the raw ALSA path, so the
  card runs at 48k and audio is resampled to the 16k the models expect (an exact /3 step).

## Credits

This project is built on these open-source projects and their documentation:

- [Silero VAD](https://github.com/snakers4/silero-vad) — voice-activity detection
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper on CTranslate2
- [Piper](https://github.com/rhasspy/piper) — neural text-to-speech
- [Ollama](https://github.com/ollama/ollama) and [ollama-python](https://github.com/ollama/ollama-python) — local LLM serving
- [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) — audio I/O
- [SciPy](https://scipy.org/) — polyphase resampling
