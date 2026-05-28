from datetime import datetime
from collections import defaultdict
import pandas as pd


class SessionTracker:
    """Logs per-frame attention data and generates session reports."""

    def __init__(self):
        self.start_time  = datetime.now()
        self.logs        = []          # list of dicts per frame
        self.state_counts = defaultdict(int)

    # ── Logging ───────────────────────────────────────────────
    def log(self, state: str, ear: float, gaze: float, score: int):
        self.state_counts[state] += 1
        self.logs.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "state":     state,
            "ear":       ear,
            "gaze":      gaze,
            "score":     score
        })

    # ── Report ────────────────────────────────────────────────
    def get_report(self) -> dict:
        total = len(self.logs)
        if total == 0:
            return {
                "total_frames": 0,
                "session_duration_s": 0,
                "focused_%": 0,
                "distracted_%": 0,
                "drowsy_%": 0,
                "avg_attention_score": 0,
                "avg_EAR": 0,
                "avg_gaze_deviation": 0,
            }

        elapsed = (datetime.now() - self.start_time).seconds
        scores  = [l["score"] for l in self.logs]
        ears    = [l["ear"]   for l in self.logs]
        gazes   = [l["gaze"]  for l in self.logs]

        def pct(state):
            return round(self.state_counts[state] / total * 100, 1)

        return {
            "total_frames":          total,
            "session_duration_s":    elapsed,
            "focused_%":             pct("focused"),
            "distracted_%":          pct("distracted"),
            "drowsy_%":              pct("drowsy"),
            "avg_attention_score":   round(sum(scores) / total, 1),
            "avg_EAR":               round(sum(ears)   / total, 3),
            "avg_gaze_deviation":    round(sum(gazes)  / total, 3),
        }

    def get_logs_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)
