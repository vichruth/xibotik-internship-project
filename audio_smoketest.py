"""Audio check: play a test tone, record three seconds, report the level, and play the
recording back through the 16 kHz round-trip. Confirms the mic and speaker work before
running the full assistant."""
import sys
import numpy as np
import sounddevice as sd

import config
import audio_io


def main():
    in_dev = audio_io.find_device(config.INPUT_DEVICE_MATCH, "input")
    out_dev = audio_io.find_device(config.OUTPUT_DEVICE_MATCH, "output")
    dr = config.DEVICE_SAMPLE_RATE
    print(f"input  device {in_dev}: {sd.query_devices(in_dev)['name']}")
    print(f"output device {out_dev}: {sd.query_devices(out_dev)['name']}")

    player = audio_io.Player(out_dev)
    t = np.linspace(0, 0.4, int(dr * 0.4), endpoint=False)
    tone = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print("playing 440 Hz test tone...")
    player.play(tone)
    player.wait_idle(timeout=3)

    print("recording 3s - say something now...")
    rec = sd.rec(int(dr * 3), samplerate=dr, channels=1, dtype="float32", device=in_dev)
    sd.wait()
    rec = rec[:, 0]
    rms = float(np.sqrt(np.mean(rec ** 2)))
    print(f"captured: RMS={rms:.4f}  ({'signal OK' if rms > 0.005 else 'too quiet - check mic'})")

    r16 = audio_io.down_to_16k(rec)
    print(f"resampled 48k({len(rec)}) -> 16k({len(r16)}) samples")
    back = audio_io.up_to_device(r16, config.SAMPLE_RATE)

    print("playing the recording back...")
    player.play(back)
    player.wait_idle(timeout=5)
    player.close()
    print("done.")
    return 0 if rms > 0.005 else 1


if __name__ == "__main__":
    sys.exit(main())
