"""Text-to-speech with Piper, en_US-lessac-medium, on CPU.

Piper (https://github.com/rhasspy/piper) is a small VITS model. On CPU it synthesizes a
short sentence in roughly 50-100 ms, which is fast enough to drive sentence-by-sentence
streaming. Keeping it on CPU leaves the whole GPU for STT and the LLM. Piper renders at
22050 Hz; output is resampled to the 48k device rate.
"""
import glob
import os

import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

import config
import audio_io


def _find_voice():
    onnx = glob.glob(os.path.join(config.TTS_MODEL_DIR, f"{config.TTS_VOICE}*.onnx"))
    if not onnx:
        raise RuntimeError(
            f"voice {config.TTS_VOICE} not found in {config.TTS_MODEL_DIR}/. "
            f"Download it with: python -m piper.download_voices {config.TTS_VOICE} "
            f"--download-dir {config.TTS_MODEL_DIR}"
        )
    return onnx[0]


class TTS:
    def __init__(self):
        self.voice = PiperVoice.load(_find_voice(), use_cuda=not config.TTS_ON_CPU)
        self.sr = self.voice.config.sample_rate          # 22050 for lessac-medium
        self._syn = SynthesisConfig(normalize_audio=False)   # skip the loudness pass to save a little time

    def synth_native(self, text):
        """Synthesize text to float32 mono at the voice's native rate."""
        chunks = [ch.audio_int16_array.astype(np.float32) / 32768.0
                  for ch in self.voice.synthesize(text, syn_config=self._syn)]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def synth_48k(self, text):
        """Synthesize text to float32 mono at the 48k device rate, ready for playback."""
        return audio_io.up_to_device(self.synth_native(text), self.sr)

    def warmup(self):
        self.synth_native("ready")
