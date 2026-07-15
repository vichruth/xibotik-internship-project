"""Tunable parameters for the voice pipeline, kept in one place."""

# --- Audio ---
# Silero VAD and Whisper both operate on 16 kHz mono, so all processing is 16k.
# The onboard ALC257 codec (raw ALSA hw path, no PipeWire resampler exposed to PortAudio)
# only accepts 44.1k/48k, so the sound card is driven at 48k and resampled in software.
# 48k -> 16k is an integer /3 decimation, which resample_poly handles exactly.
SAMPLE_RATE = 16000
DEVICE_SAMPLE_RATE = 48000
DECIM = DEVICE_SAMPLE_RATE // SAMPLE_RATE            # 3

# 32 ms blocks: 512 samples at 16k is the frame size the Silero model expects.
BLOCK_MS = 32
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000       # 512  (16k)
DEVICE_BLOCK_SAMPLES = DEVICE_SAMPLE_RATE * BLOCK_MS // 1000   # 1536 (48k)

# Match the codec by name; ALSA device indices are not stable across reboots.
INPUT_DEVICE_MATCH = "ALC257"
OUTPUT_DEVICE_MATCH = "ALC257"

# --- VAD (Silero) ---
VAD_THRESHOLD = 0.5          # speech probability above which a frame is counted as speech
VAD_MIN_SILENCE_MS = 700     # trailing silence that ends a turn; survives mid-sentence pauses
VAD_MIN_SPEECH_MS = 200      # discard shorter blips (coughs, clicks, key presses)
VAD_SPEECH_PAD_MS = 200      # audio kept before speech onset so the first phoneme isn't clipped

# --- STT (faster-whisper) ---
STT_MODEL = "small.en"       # small.en trades a little accuracy for latency; base.en misses
                             # technical words, medium is too slow on a 6 GB laptop GPU
STT_COMPUTE_TYPE = "int8_float16"   # int8 weights / fp16 compute; small enough to share VRAM with the LLM
STT_DEVICE = "cuda"

# --- LLM (Ollama) ---
LLM_MODEL = "qwen2.5:3b"     # 3B keeps time-to-first-token low; 7B does not fit the 2 s budget here
LLM_HOST = "http://127.0.0.1:11434"
LLM_NUM_PREDICT = 200        # spoken replies are short; cap generation length
LLM_KEEP_ALIVE = "30m"       # keep the model resident in VRAM between turns (avoids cold reloads)
SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. "
    "Reply in one to three short spoken sentences. "
    "No markdown, no lists, no emoji, no stage directions - your text is read aloud."
)

# --- TTS (Piper) ---
TTS_VOICE = "en_US-lessac-medium"
TTS_MODEL_DIR = "voices"
TTS_ON_CPU = True            # run TTS on CPU so the GPU stays free for STT + LLM

# --- Fallback timers ---
WATCHDOG_STALL_MS = 1500     # no first token by here -> play a short "give me a second" line
WATCHDOG_DEGRADE_MS = 5000   # still nothing by here -> honest holding reply, reopen the mic
