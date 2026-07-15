"""LLM responses from qwen2.5:3b served locally by Ollama, streamed token by token.

A 3B model keeps time-to-first-token low enough to stay inside the latency budget on a
6 GB GPU. Streaming lets the pipeline start speaking the first sentence while the rest is
still being generated (see main.py). Uses the official ollama-python client.
"""
import ollama

import config


class LLMUnavailable(Exception):
    """Raised when Ollama can't be reached or the model can't be loaded.
    The caller degrades gracefully instead of surfacing an error to the user."""


class LLM:
    def __init__(self):
        self.client = ollama.Client(host=config.LLM_HOST)
        self.history = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    def warmup(self):
        """Load the model into VRAM ahead of the first turn so token one isn't a cold start."""
        try:
            list(self.client.generate(
                model=config.LLM_MODEL, prompt="hi", stream=True,
                keep_alive=config.LLM_KEEP_ALIVE, options={"num_predict": 1},
            ))
        except Exception as e:
            raise LLMUnavailable(str(e))

    def stream(self, user_text, cancel=None):
        """Yield response text chunks for user_text.

        If `cancel` is set mid-generation (barge-in) it stops and rolls the turn back out of
        history. The completed reply is appended to history only if generation finishes.
        Raises LLMUnavailable if the server can't be reached.
        """
        self.history.append({"role": "user", "content": user_text})
        acc = []
        try:
            stream = self.client.chat(
                model=config.LLM_MODEL,
                messages=self.history,
                stream=True,
                keep_alive=config.LLM_KEEP_ALIVE,
                options={"num_predict": config.LLM_NUM_PREDICT, "temperature": 0.7},
            )
            for chunk in stream:
                if cancel is not None and cancel.is_set():
                    self.history.pop()          # discard the interrupted turn
                    return
                tok = chunk["message"]["content"]
                if tok:
                    acc.append(tok)
                    yield tok
        except Exception as e:
            self.history.pop()
            raise LLMUnavailable(str(e))
        self.history.append({"role": "assistant", "content": "".join(acc)})
