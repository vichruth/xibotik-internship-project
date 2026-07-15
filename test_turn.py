"""Headless turn test: exercise STT + LLM + TTS + sentence streaming + playback on fixed
transcripts, without a microphone. Useful for checking latency and GPU memory in isolation."""
import subprocess
import time

import metrics as metrics_mod
from main import Assistant


def gpu_mem():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    ).decode().strip()
    return out


def main():
    a = Assistant()
    a.warmup()
    print("GPU memory with all models loaded:", gpu_mem(), "MiB (used/total)")

    for transcript in ["What is a humanoid robot?", "Give me one tip for staying focused."]:
        print(f"\n[sim] You: {transcript}")
        turn = metrics_mod.Turn(time.perf_counter())
        turn.mark("t_ack")          # no filler in the headless path
        turn.mark("t_stt_done")     # text is supplied directly, so STT is skipped
        a.cancel.clear()
        a._run_turn(transcript, turn)
        a.player.wait_idle(timeout=15)

    print("\nGPU memory at end:", gpu_mem(), "MiB")
    print(a.metrics.summary())
    a.player.close()
    a.in_stream.close()


if __name__ == "__main__":
    main()
