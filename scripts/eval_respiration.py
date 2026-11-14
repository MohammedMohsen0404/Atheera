"""Evaluate breathing-rate estimation on WiRe-Breath against the reference respiration signal.

Classic (non-learned) estimator only; writes results/eval_respiration_classic.json. The headline numbers in
results/eval_respiration.json come from scripts/train_breath_confidence.py (fused amplitude + phase estimator with tracking).
Sliding windows (default 20 s, hop 5 s) inside each 30 s recording; reference BR from gt_signal over the same span.
Reports MAE, share within ±2 br/min, and confidence-gated MAE + coverage, grouped by environment and posture.

  python scripts/eval_respiration.py --root data/raw/wirebreath
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.ingest import wirebreath as WB  # noqa: E402
from atheer.modules import respiration as R  # noqa: E402
from atheer.fusion.engine import CONF_MIN  # noqa: E402


def stats(err, conf):
    err, conf = np.asarray(err), np.asarray(conf)
    ok = conf >= CONF_MIN
    return {"n": int(err.size), "mae": float(np.mean(err)), "median_ae": float(np.median(err)),
            "within2": float(np.mean(err <= 2)),
            "gated_mae": float(np.mean(err[ok])) if ok.any() else None,
            "gated_within2": float(np.mean(err[ok] <= 2)) if ok.any() else None,
            "coverage": float(ok.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "wirebreath")
    ap.add_argument("--win", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = WB.find_files(a.root)
    if a.limit:
        files = files[:a.limit]
    rows = []
    for k, f in enumerate(files):
        rec = WB.load(f)
        fs, amp = rec.fs, rec.amp
        n, h = int(a.win * fs), int(a.hop * fs)
        xf = R.preprocess(amp, fs)  # filter the whole recording once, then cut windows
        for end in range(n, xf.shape[0] + 1, h):
            e = R.estimate_classic(xf[end - n:end], fs)
            t0, t1 = rec.time[end - n], rec.time[end - 1]
            ref = R.reference_rate(rec.gt_time, rec.gt_signal, t0, t1)
            if not (np.isfinite(ref) and np.isfinite(e.bpm)):
                continue
            rows.append({**rec.meta, "est": e.bpm, "ref": ref, "conf": e.confidence, "err": abs(e.bpm - ref)})
        if k % 50 == 0:
            print(f"{k}/{len(files)} files, {len(rows)} windows")
    if not rows:
        sys.exit("no windows evaluated")
    groups = defaultdict(list)
    for r in rows:
        groups["All"].append(r)
        groups[f"Environment: {r.get('environment', '?')}"].append(r)
        groups[f"Posture: {r.get('posture', '?')}"].append(r)
        if "room" in r:
            groups[f"{r['room']}"].append(r)
    out = {"n_files": len(files), "n_windows": len(rows), "window_s": a.win, "hop_s": a.hop,
           "conf_min": CONF_MIN, "reference": "spectral peak of WiRe-Breath gt_signal over the same window",
           "groups": {g: stats([r["err"] for r in rs], [r["conf"] for r in rs]) for g, rs in sorted(groups.items())}}
    # confidence calibration: error by confidence decile
    conf = np.array([r["conf"] for r in rows]); err = np.array([r["err"] for r in rows])
    edges = np.quantile(conf, np.linspace(0, 1, 6))
    out["by_confidence"] = [{"lo": float(lo), "hi": float(hi), "mae": float(err[(conf >= lo) & (conf <= hi)].mean())}
                            for lo, hi in zip(edges[:-1], edges[1:])]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval_respiration_classic.json").write_text(json.dumps(out, indent=1))
    for g, s in out["groups"].items():
        print(f"{g:32s} n={s['n']:5d}  MAE={s['mae']:.2f}  ±2={s['within2']:.0%}  gated MAE={s['gated_mae']}  cov={s['coverage']:.0%}")
    print("by confidence:", out["by_confidence"])


if __name__ == "__main__":
    main()
