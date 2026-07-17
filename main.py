"""Offline real-time voice assistant - main loop and turn orchestration.

Pipeline:  mic -> VAD endpoint -> filler ack -> STT -> LLM (streamed)
           -> sentence split -> Piper -> speaker,  with a watchdog and barge-in.

The first reply sentence is spoken as soon as it is complete, while the LLM is still
generating the rest, so the user hears audio well before the full response is ready.
"""
import queue
import re
import threading
import time

import sounddevice as sd

import config
import audio_io
import metrics as metrics_mod
from state import S, Fallback
from vad import StreamingVAD
from stt import STT
from llm import LLM, LLMUnavailable
from tts import TTS

# Sentence boundary: end punctuation, or a comma once a clause is long enough. The comma
# rule ships the first chunk sooner when the model hasn't reached a full stop yet.
_HARD = re.compile(r"[.!?\n]")


def split_sentences(buf):
    """Split completed sentences off the front of `buf`. Returns (sentences, remainder)."""
    out, last = [], 0
    for m in _HARD.finditer(buf):
        out.append(buf[last:m.end()].strip())
        last = m.end()
    remainder = buf[last:]
    if len(remainder.split()) >= 12 and "," in remainder:
        i = remainder.rfind(",")
        out.append(remainder[:i + 1].strip())
        remainder = remainder[i + 1:]
    return [s for s in out if s], remainder


class Assistant:
    def __init__(self):
        print("loading models...")
        self.metrics = metrics_mod.Metrics()
        self.vad = StreamingVAD()
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.fallback = Fallback()

        in_dev = audio_io.find_device(config.INPUT_DEVICE_MATCH, "input")
        out_dev = audio_io.find_device(config.OUTPUT_DEVICE_MATCH, "output")
        self.player = audio_io.Player(out_dev)
        self.in_q = queue.Queue()
        self.in_stream = sd.InputStream(
            samplerate=config.DEVICE_SAMPLE_RATE, channels=1, dtype="float32",
            device=in_dev, blocksize=config.DEVICE_BLOCK_SAMPLES, callback=self._in_cb,
        )

        self.state = S.IDLE
        self.cancel = threading.Event()      # set to interrupt the active turn (barge-in / hard stall)
        self.worker = None
        self.ollama_ok = True

    def _in_cb(self, indata, frames, time_info, status):
        # runs on the PortAudio thread; just hand the block off to the main loop
        self.in_q.put(indata[:, 0].copy())

    def warmup(self):
        print("warming up STT + LLM + TTS...")
        self.stt.warmup()
        try:
            self.llm.warmup()
        except LLMUnavailable as e:
            self.ollama_ok = False
            print(f"  ! Ollama warmup failed ({e}); will run in degraded mode.")
        self.tts.warmup()
        print("pre-rendering fallback phrases...")
        self.fallback.prerender(self.tts)
        print("ready. speak when you like. (Ctrl+C to quit)\n")

    def _run_turn(self, transcript, turn):
        """Generate and speak a reply. Runs on a worker thread so the main loop stays free
        to watch the mic for barge-in."""
        first_token = threading.Event()
        degraded = threading.Event()

        def watchdog():
            # first escalation: a stall line if the model is slow to start
            if not first_token.wait(config.WATCHDOG_STALL_MS / 1000):
                if self.cancel.is_set():
                    return
                self.player.play(self.fallback.random_stall())
            # second escalation: give up on this turn, apologise, reopen the mic
            remaining = (config.WATCHDOG_DEGRADE_MS - config.WATCHDOG_STALL_MS) / 1000
            if not first_token.wait(remaining):
                if self.cancel.is_set():
                    return
                degraded.set()
                self.cancel.set()
                self.player.play(self.fallback.random_degrade())

        threading.Thread(target=watchdog, daemon=True).start()

        buf = ""
        spoke = False
        try:
            for tok in self.llm.stream(transcript, cancel=self.cancel):
                if not first_token.is_set():
                    turn.mark("t_llm_first_token")
                    first_token.set()
                buf += tok
                sentences, buf = split_sentences(buf)
                for s in sentences:
                    if self.cancel.is_set():
                        break
                    audio = self.tts.synth_48k(s)
                    if not spoke:
                        turn.mark("t_first_audio")
                        spoke = True
                    self.player.play(audio)
            # speak whatever is left over (a final clause with no trailing punctuation)
            tail = buf.strip()
            if tail and not self.cancel.is_set():
                audio = self.tts.synth_48k(tail)
                if not spoke:
                    turn.mark("t_first_audio")
                    spoke = True
                self.player.play(audio)
        except LLMUnavailable as e:
            self.ollama_ok = False
            first_token.set()   # release the watchdog
            if not self.cancel.is_set():
                self.player.play(self.fallback.random_degrade())
            print(f"  ! LLM unavailable ({e}); degraded response.")

        first_token.set()
        turn.mark("t_response_end")
        if not degraded.is_set() and not self.cancel.is_set():
            self.metrics.record(turn, transcript)
        self.state = S.LISTENING

    def run(self):
        self.in_stream.start()
        self.state = S.LISTENING
        barge_frames = 0
        prev_speaking = False
        speak_start = 0.0
        # Barge-in guards against the assistant's own voice leaking into the mic on laptop
        # speakers: a higher probability threshold than endpointing, a short run of frames,
        # and a grace window right after playback starts. Headphones avoid the echo entirely.
        BARGE_CONFIRM = 10           # ~320 ms of sustained speech
        BARGE_THRESHOLD = 0.95
        BARGE_GRACE_S = 0.8
        try:
            while True:
                block48k = self.in_q.get()
                frame16k = audio_io.down_to_16k(block48k)

                speaking = self.player.is_active() or (self.worker is not None and self.worker.is_alive())

                if speaking:
                    if not prev_speaking:
                        speak_start = time.perf_counter()
                    prev_speaking = True
                    in_grace = (time.perf_counter() - speak_start) < BARGE_GRACE_S
                    if not in_grace and self.vad.speech_prob(frame16k) >= BARGE_THRESHOLD:
                        barge_frames += 1
                    else:
                        barge_frames = 0
                    if barge_frames >= BARGE_CONFIRM:
                        print("  (barge-in: user interrupted)")
                        self.cancel.set()
                        self.player.barge_in()
                        if self.worker:
                            self.worker.join(timeout=1.0)
                        self.vad.reset()
                        self.cancel.clear()
                        barge_frames = 0
                        self.state = S.LISTENING
                    continue

                # playback just ended: clear the VAD so the echo tail isn't read as new speech
                if prev_speaking:
                    prev_speaking = False
                    barge_frames = 0
                    self.vad.reset()

                event = self.vad.process(frame16k, block48k)
                if isinstance(event, tuple) and event[0] == "utterance":
                    turn = metrics_mod.Turn(time.perf_counter())   # t0 = end of speech
                    utt16k = audio_io.down_to_16k(event[1])

                    # acknowledge immediately, in parallel with STT and the LLM
                    self.player.play(self.fallback.random_filler())
                    turn.mark("t_ack")

                    self.state = S.THINKING
                    transcript = self.stt.transcribe(utt16k)
                    turn.mark("t_stt_done")
                    if not transcript:
                        self.state = S.LISTENING
                        continue
                    print(f"\nYou: {transcript}")

                    self.state = S.SPEAKING
                    self.cancel.clear()
                    self.worker = threading.Thread(
                        target=self._run_turn, args=(transcript, turn), daemon=True)
                    self.worker.start()
        except KeyboardInterrupt:
            print("\n\n" + self.metrics.summary())
        finally:
            self.in_stream.stop()
            self.in_stream.close()
            self.player.close()


if __name__ == "__main__":
    assistant = Assistant()
    assistant.warmup()
    assistant.run()
