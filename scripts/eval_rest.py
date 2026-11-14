"""Sanity check for the rest / sleep-period tracker on real CSI.

WiRe-Breath subjects lie or sit still for every recording, so a correct tracker should score nearly all of that
time as rest. This measures the false-awake rate (still people scored as moving), and whether a second person
moving nearby (the 'Interference' recordings) is wrongly counted as the subject waking. It cannot measure the
opposite error (awake people scored as resting): no public CSI dataset pairs breathing with labelled wake periods.

Each WiRe-Breath track (results/tracks/wb_*.json, real recordings replayed per second) is run through the same
anomaly detector and fusion code as the demo, as if the whole track were a sleep period.

  python scripts/eval_rest.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.fusion.engine import Frame, breath_ok, external_motion  # noqa: E402
from atheer.modules.anomaly import AnomalyDetector  # noqa: E402
from atheer.modules.rest import RestTracker  # noqa: E402


def run_track(tr: dict) -> dict:
    det, rest = AnomalyDetector(), RestTracker()
    for i in range(len(tr["t"])):
        b = tr["bpm"][i]
        f = Frame(t=i, crew_id="X", module="CQ", scheduled="SLEEP", bpm=math.nan if b is None else b,
                  br_conf=tr["conf"][i] or 0.0, motion=tr["motion"][i] or 0.0, activity="lie_down")
        r = det.update(f.motion, 0.0, f.bpm, f.br_conf)
        f.anomaly = r.score
        rest.update(True, breath_ok(f), f.anomaly, f.activity, external=external_motion(f))
    return rest.summary()


def main():
    rows = []
    for p in sorted((ROOT / "results" / "tracks").glob("wb_*.json")):
        tr = json.loads(p.read_text())
        s = run_track(tr)
        rows.append({"track": tr["id"], **tr["meta"], **s})
    out = {"protocol": "every WiRe-Breath track replayed as a sleep period through the demo's anomaly + fusion "
                       "code; subjects were still throughout, so time not scored as rest is a false-awake error",
           "limits": "cannot measure awake-scored-as-rest: no public CSI data pairs breathing with labelled wake",
           "groups": {}, "tracks": rows}
    groups = defaultdict(list)
    for r in rows:
        groups["All"].append(r); groups[f"Environment: {r['environment']}"].append(r)
    for g, rs in groups.items():
        per = sum(r["sleep_period_s"] for r in rs)
        out["groups"][g] = {
            "tracks": len(rs), "seconds": per,
            "rest_pct": round(100 * sum(r["rest_s"] for r in rs) / per, 1),
            "awake_pct": round(100 * sum(r["awake_s"] for r in rs) / per, 1),
            "unclear_pct": round(100 * sum(r["unclear_s"] for r in rs) / per, 1),
            "nearby_motion_pct": round(100 * sum(r["nearby_motion_s"] for r in rs) / per, 1),
            "false_movements_per_hour": round(3600 * sum(r["movements"] for r in rs) / per, 1),
            "tracks_below_90pct_rest": [r["track"] for r in rs if (r["rest_pct"] or 0) < 90],
        }
    (ROOT / "results" / "eval_rest.json").write_text(json.dumps(out, indent=1))
    for g, s in out["groups"].items():
        print(f"{g:28s} {s}")


if __name__ == "__main__":
    main()
