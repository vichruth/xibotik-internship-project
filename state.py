"""Conversation states and the canned fallback phrases.

State flow:  IDLE -> LISTENING -> THINKING -> SPEAKING -> (barge-in) -> LISTENING

Three fallback behaviours keep the conversation feeling responsive when the model is slow
or unavailable:
  1. Instant acknowledgement - a short filler plays the moment the user stops speaking,
     in parallel with STT and the LLM starting, so there is never dead air.
  2. Watchdog - a stall line if the first token is late, then an honest holding reply that
     reopens the mic. It never shows an error and never ends the conversation.
  3. Barge-in - handled in main.py: if the user speaks over the assistant, playback is
     flushed and generation is cancelled.

All phrases are synthesized to audio once at startup so playing them at runtime is free.
Keeping the wording here makes it easy to review and adjust.
"""
import random
from enum import Enum, auto

import numpy as np


class S(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


# Short acknowledgements played immediately when the user stops speaking.
FILLERS = ["Mhm.", "Right, so.", "Let me think.", "Okay.", "Good question.", "Sure.", "Got it."]

# Played if the first token is late but generation is still expected.
STALLS = [
    "That's an interesting one, give me a second.",
    "Hmm, let me get that right for you.",
    "One moment, thinking that through.",
]

# Played on a hard stall or when the model is unavailable. Honest, and hands the turn back.
DEGRADES = [
    "Sorry, that's taking me longer than I'd like. Could you say it once more?",
    "I'm having a slow moment here - mind repeating that for me?",
    "Let me reset on that one. Go ahead and ask me again.",
]


class Fallback:
    """Pre-rendered filler / stall / degrade audio at 48 kHz, built once at startup."""

    def __init__(self):
        self.fillers = []
        self.stalls = []
        self.degrades = []

    def prerender(self, tts):
        self.fillers = [tts.synth_48k(t) for t in FILLERS]
        self.stalls = [tts.synth_48k(t) for t in STALLS]
        self.degrades = [tts.synth_48k(t) for t in DEGRADES]

    def random_filler(self):
        return random.choice(self.fillers) if self.fillers else np.zeros(0, np.float32)

    def random_stall(self):
        return random.choice(self.stalls) if self.stalls else np.zeros(0, np.float32)

    def random_degrade(self):
        return random.choice(self.degrades) if self.degrades else np.zeros(0, np.float32)
