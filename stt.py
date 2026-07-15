"""Speech-to-text with faster-whisper (small.en, int8_float16, CUDA).

faster-whisper (https://github.com/SYSTRAN/faster-whisper) runs Whisper on the CTranslate2
engine, which is several times faster than the reference implementation. The int8_float16
compute type keeps the model small enough to share the 6 GB GPU with the LLM.

Whisper is known to emit confident phrases ("thank you", "please subscribe", etc.) on
near-silence. The turn is gated on input energy first, and low-confidence / high no-speech
segments are dropped, so a hallucinated transcript never reaches the LLM. The thresholds
mirror the openai-whisper defaults exposed by faster-whisper.
"""
import numpy as np
from faster_whisper import WhisperModel

import config


class STT:
    def __init__(self):
        self.model = WhisperModel(
            config.STT_MODEL,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )

    def transcribe(self, audio16k):
        """Transcribe 16 kHz mono float32. Returns the text, or '' if the audio looks like noise."""
        if float(np.sqrt(np.mean(audio16k ** 2))) < 0.01:
            return ""   # too quiet to be speech; skip the model entirely

        # greedy decode: a single spoken turn is short, so beam search is not worth the latency.
        # our own VAD already segmented the audio, so Whisper's internal VAD is left off.
        segments, _ = self.model.transcribe(
            audio16k,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        kept = [s.text for s in segments
                if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
        return "".join(kept).strip()

    def warmup(self):
        """Run one inference so CUDA context setup happens before the first real turn."""
        self.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32))
