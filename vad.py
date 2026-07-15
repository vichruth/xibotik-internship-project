"""Streaming voice-activity detection and endpointing with Silero VAD.

Silero (https://github.com/snakers4/silero-vad) is a small neural VAD; it is far less
twitchy on background noise than energy- or WebRTC-based detectors, and runs comfortably
on CPU (well under a millisecond per frame), leaving the GPU for STT and the LLM.

The model expects 512-sample frames at 16 kHz. Detection runs on the 16k frame, but the
raw 48k audio is accumulated for the utterance so it can be decimated once at the end -
decimating each block independently would leave periodic filter artefacts at block edges.

process(frame16k, frame48k) returns:
  None                       - no state change
  "speech_start"             - onset of speech (used to detect barge-in)
  ("utterance", np.ndarray)  - a completed utterance, raw 48k
"""
import numpy as np
import torch
from silero_vad import load_silero_vad

import config

torch.set_num_threads(1)   # single frame at a time; extra threads only add overhead


class StreamingVAD:
    def __init__(self):
        self.model = load_silero_vad(onnx=False)
        self.reset()
        self._pad_frames = max(1, config.VAD_SPEECH_PAD_MS // config.BLOCK_MS)
        self._min_speech_frames = max(1, config.VAD_MIN_SPEECH_MS // config.BLOCK_MS)
        self._min_silence_frames = max(1, config.VAD_MIN_SILENCE_MS // config.BLOCK_MS)

    def reset(self):
        self.model.reset_states()
        self._triggered = False
        self._speech = []          # accumulated utterance frames (raw 48k)
        self._pad = []             # rolling pre-onset pad (raw 48k)
        self._silence_run = 0
        self._speech_frames = 0

    def speech_prob(self, frame16k):
        """Speech probability for one frame, without touching the endpointing state.
        Used to detect barge-in while the assistant is speaking."""
        return self.model(torch.from_numpy(frame16k), config.SAMPLE_RATE).item()

    def process(self, frame16k, frame48k):
        prob = self.model(torch.from_numpy(frame16k), config.SAMPLE_RATE).item()
        is_speech = prob >= config.VAD_THRESHOLD
        event = None

        if not self._triggered:
            self._pad.append(frame48k)
            if len(self._pad) > self._pad_frames:
                self._pad.pop(0)
            if is_speech:
                self._triggered = True
                self._speech = list(self._pad)   # prepend the pad so the onset isn't clipped
                self._pad = []
                self._silence_run = 0
                self._speech_frames = 1
                event = "speech_start"
        else:
            self._speech.append(frame48k)
            if is_speech:
                self._silence_run = 0
                self._speech_frames += 1
            else:
                self._silence_run += 1
                if self._silence_run >= self._min_silence_frames:
                    audio = np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)
                    long_enough = self._speech_frames >= self._min_speech_frames
                    self.reset()
                    if long_enough:
                        event = ("utterance", audio)
                    # otherwise it was too short to be real speech; drop it
        return event
