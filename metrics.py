"""Per-turn latency measurement.

The stage of interest is TTFA (time to first audio) - how long after the user stops
speaking they hear the start of a reply. Every stage is timestamped relative to end of
speech, printed per turn, and appended to a CSV for later analysis.
"""
import csv
import os
import time
from statistics import median

# stages in pipeline order; end of speech is the zero reference
STAGES = ["t_ack", "t_stt_done", "t_llm_first_token", "t_first_audio", "t_response_end"]
LABELS = {
    "t_ack": "ack",
    "t_stt_done": "stt",
    "t_llm_first_token": "llm_1st_tok",
    "t_first_audio": "TTFA",
    "t_response_end": "resp_end",
}


class Turn:
    """A single user/assistant exchange. mark() records a stage time relative to end of speech."""

    def __init__(self, t_vad_end=None):
        self.t0 = t_vad_end if t_vad_end is not None else time.perf_counter()
        self.marks = {}

    def mark(self, stage):
        self.marks[stage] = time.perf_counter() - self.t0
        return self.marks[stage]

    def ms(self, stage):
        return self.marks.get(stage)


class Metrics:
    def __init__(self, csv_path="latency_log.csv"):
        self.csv_path = csv_path
        self.turns = []
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["turn", "transcript"] + STAGES)

    def record(self, turn, transcript=""):
        self.turns.append(turn)
        row = [len(self.turns), transcript[:60]]
        row += [f"{turn.ms(s):.3f}" if turn.ms(s) is not None else "" for s in STAGES]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._print_turn(turn)

    def _print_turn(self, turn):
        parts = []
        for s in STAGES:
            v = turn.ms(s)
            parts.append(f"{LABELS[s]}={v*1000:.0f}ms" if v is not None else f"{LABELS[s]}=-")
        ttfa = turn.ms("t_first_audio")
        flag = "  <-- TTFA under 2s" if (ttfa is not None and ttfa < 2.0) else ""
        print("  [latency] " + "  ".join(parts) + flag)

    def summary(self):
        """Median for each stage across the session."""
        if not self.turns:
            return "no turns recorded."
        lines = [f"session: {len(self.turns)} turns", "  median per stage (ms from end of speech):"]
        for s in STAGES:
            vals = [t.ms(s) for t in self.turns if t.ms(s) is not None]
            if vals:
                lines.append(f"    {LABELS[s]:12s} {median(vals)*1000:7.0f}   (n={len(vals)})")
        return "\n".join(lines)
