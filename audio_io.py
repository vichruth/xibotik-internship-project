"""Audio device lookup, sample-rate conversion, and a streaming output player.

The sound card runs at 48 kHz; the pipeline works at 16 kHz. Conversion uses polyphase
resampling (scipy.signal.resample_poly), which applies an anti-aliasing filter.

The callback-based OutputStream follows the python-sounddevice streaming pattern
(https://python-sounddevice.readthedocs.io). Playback is fed from a queue so synthesized
sentences play back to back, and the queue can be flushed mid-stream for barge-in.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

import config


def find_device(match, kind):
    """Return the index of the first device whose name contains `match` and has the
    requested channel direction ('input' or 'output')."""
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if match.lower() in d["name"].lower() and ch > 0:
            return i
    raise RuntimeError(f"no {kind} device matching {match!r}")


def down_to_16k(block48k):
    """48k mono float32 -> 16k mono float32 (integer /3 decimation)."""
    return resample_poly(block48k, 1, config.DECIM).astype(np.float32)


def up_to_device(audio, src_rate):
    """Resample mono float32 from src_rate up to the 48k device rate."""
    if src_rate == config.DEVICE_SAMPLE_RATE:
        return audio.astype(np.float32)
    g = np.gcd(src_rate, config.DEVICE_SAMPLE_RATE)
    return resample_poly(audio, config.DEVICE_SAMPLE_RATE // g, src_rate // g).astype(np.float32)


class Player:
    """Queue-fed streaming output at 48 kHz. Chunks play back to back; barge_in() drops
    whatever is still queued so playback stops immediately."""

    def __init__(self, device):
        self.device = device
        self._q = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._active = threading.Event()
        self.stream = sd.OutputStream(
            samplerate=config.DEVICE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=config.DEVICE_BLOCK_SAMPLES,
            callback=self._cb,
        )
        self.stream.start()

    def _cb(self, outdata, frames, time_info, status):
        need = frames
        out = self._buf
        while len(out) < need:
            try:
                out = np.concatenate([out, self._q.get_nowait()])
            except queue.Empty:
                break
        if len(out) >= need:
            outdata[:, 0] = out[:need]
            self._buf = out[need:]
        else:
            # underrun: emit what we have, pad the rest with silence
            outdata[:len(out), 0] = out
            outdata[len(out):, 0] = 0.0
            self._buf = np.zeros(0, dtype=np.float32)
        if self._q.empty() and len(self._buf) == 0:
            self._active.clear()

    def play(self, audio48k):
        with self._lock:
            self._active.set()
            self._q.put(np.asarray(audio48k, dtype=np.float32))

    def is_active(self):
        return self._active.is_set()

    def barge_in(self):
        """Flush the queue and the partial buffer so playback stops on the next callback."""
        with self._lock:
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._buf = np.zeros(0, dtype=np.float32)
            self._active.clear()

    def wait_idle(self, timeout=None):
        t0 = time.time()
        while self._active.is_set():
            if timeout and time.time() - t0 > timeout:
                return False
            time.sleep(0.01)
        return True

    def close(self):
        self.stream.stop()
        self.stream.close()
