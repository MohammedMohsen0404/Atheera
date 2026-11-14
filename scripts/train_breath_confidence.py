"""Evaluate the breathing estimator on WiRe-Breath and fit its confidence calibration.

Rate: no learning (fused amplitude + cross-subcarrier-ratio spectrum, Viterbi tracking over spectral peaks).
Confidence: isotonic map from peak prominence to P(reading within ±2 br/min), times a coherence guard.
The reported confidences are cross-fitted by subject (each person's from a calibration fitted on the others).

Outputs
  models/breath_confidence.joblib   isotonic calibration fitted on all subjects
  results/eval_respiration.json     metrics for the shipped method and every comparison variant

  python scripts/train_breath_confidence.py [--root data/raw/wirebreath/extracted]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.ingest import wirebreath as WB  # noqa: E402
from atheer.modules import respiration as R  # noqa: E402
from atheer.fusion.engine import CONF_MIN  # noqa: E402

VARIANTS = {  # name -> (use phase channels, use tracking)
    "fused + tracking": (True, True),
    "fused, no tracking": (True, False),
    "amplitude + tracking": (False, True),
    "amplitude, no tracking": (False, False),
}
SHIPPED = "fused + tracking"


def stats(err, conf=None):
    err = np.asarray(err)
    out = {"n": int(err.size), "mae": float(err.mean()), "median_ae": float(np.median(err)),
           "within2": float(np.mean(err <= 2)), "bias": None}
    if conf is not None:
        conf = np.asarray(conf); ok = conf >= CONF_MIN
        out.update(gated_mae=float(err[ok].mean()) if ok.any() else None,
                   gated_within2=float(np.mean(err[ok] <= 2)) if ok.any() else None, coverage=float(ok.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "wirebreath")
    ap.add_argument("--win", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=5.0)
    a = ap.parse_args()
    files = WB.find_files(a.root)
    rows = []
    for k, f in enumerate(files):
        rec = WB.load(f)
        fs = rec.fs
        n, h = int(a.win * fs), int(a.hop * fs)
        ends = list(range(n, rec.csi.shape[0] + 1, h))
        refs = [R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[e - n], rec.time[e - 1]) for e in ends]
        est = {}
        for name, (phase, trk) in VARIANTS.items():
            ch = R.channels(rec.csi if phase else rec.amp, fs)
            wss = [R.window_spectrum(ch, slice(e - n, e), fs) for e in ends]
            sc = [R._peaks_only(w.fine) for w in wss]
            path = R.track(sc, R.TRACK_LAM * 5.0 / a.hop) if trk else [int(np.argmax(s)) for s in sc]
            est[name] = [(R._parabolic(w.fine, j), w) for w, j in zip(wss, path)]
        # ambiguity check (as in R.sliding): a wider band seeing half the shipped rate zeroes the confidence
        chw = R.channels(rec.csi, fs, R.WIDE_BAND_HZ)
        wsw = [R.window_spectrum(chw, slice(e - n, e), fs, grid=R.WIDE_GRID) for e in ends]
        pw = R.track([R._peaks_only(w.fine) for w in wsw], R.TRACK_LAM * 5.0 / a.hop, R.WIDE_GRID)
        wide = [R._parabolic(w.fine, j, R.WIDE_GRID) for w, j in zip(wsw, pw)]
        xa = R.preprocess(rec.amp, fs)
        for i, (e, ref) in enumerate(zip(ends, refs)):
            if not np.isfinite(ref):
                continue
            row = {"meta": rec.meta, "ref": ref, "classic": R.estimate_classic(xa[e - n:e], fs).bpm}
            for name, lst in est.items():
                bpm, w = lst[i]
                row[name] = bpm
                if name == SHIPPED:
                    row["prom"], row["guard"] = R.confidence_features(w, bpm)
                    row["ambiguous"] = abs(wide[i] / bpm - 0.5) < R.AMBIGUITY_TOL
            rows.append(row)
        if k % 50 == 0:
            print(f"{k}/{len(files)} files, {len(rows)} windows")

    ref = np.array([r["ref"] for r in rows])
    subj = np.array([r["meta"]["subject"] for r in rows])
    err = np.abs(np.array([r[SHIPPED] for r in rows]) - ref)
    prom = np.array([r["prom"] for r in rows]); guard = np.array([r["guard"] for r in rows])
    good = (err <= 2).astype(float)
    conf = np.zeros(len(rows))
    for s in np.unique(subj):
        m = subj == s
        conf[m] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(prom[~m], good[~m]).predict(prom[m])
    conf *= guard
    conf[np.array([r["ambiguous"] for r in rows])] = 0.0
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(prom, good)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"iso": iso, "feature": "peak prominence (±2.4 br/min share of in-band power)",
                 "target": "P(|error| <= 2 br/min)", "trained_on": f"WiRe-Breath, {len(rows)} windows"},
                R.CALIB_PATH)
    R.load_calibration.cache_clear()

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        m = r["meta"]
        for g in ("All", f"Environment: {m.get('environment', '?')}", f"Posture: {m.get('posture', '?')}",
                  m.get("room", "?"), f"Subject: {m.get('subject', '?')}"):
            groups[g].append(i)
    out_groups = {}
    for g, ix in sorted(groups.items()):
        st = stats(err[ix], conf[ix])
        st["bias"] = float(np.mean(np.array([rows[i][SHIPPED] for i in ix]) - ref[ix]))
        out_groups[g] = st
    comparison = {name: stats(np.abs(np.array([r[name] for r in rows]) - ref)) for name in VARIANTS}
    comparison["classic (PCA peak, amplitude)"] = stats(np.abs(np.array([r["classic"] for r in rows]) - ref))
    for name in comparison:
        env = np.array([r["meta"]["environment"] for r in rows])
        col = "classic" if name.startswith("classic") else name
        e = np.abs(np.array([r[col] for r in rows]) - ref)
        comparison[name].update(quiet_mae=float(e[env == "Quiet"].mean()), interference_mae=float(e[env == "Interference"].mean()))

    rng = np.random.default_rng(0); fs0 = 50.0; t = np.arange(int(30 * fs0)) / fs0; synth = {}
    for b in (6.5, 8, 12, 16, 22, 30, 34):   # clean breathing at any rate must come out where it is
        amp = 20 + np.outer(np.sin(2 * np.pi * b / 60 * t), rng.uniform(-1, 1, 52)) * 0.6 + rng.normal(0, 0.5, (t.size, 52))
        e = R.estimate(amp, fs0); synth[str(b)] = [round(e.bpm, 2), round(e.confidence, 2)]
    e = R.estimate(20 + rng.normal(0, 1, (t.size, 52)), fs0); synth["noise"] = [round(e.bpm, 2), round(e.confidence, 2)]

    edges = np.quantile(conf, np.linspace(0, 1, 6))
    out = {"n_files": len(files), "n_windows": len(rows), "window_s": a.win, "hop_s": a.hop, "conf_min": CONF_MIN,
           "method": "fused amplitude + cross-subcarrier-ratio phase spectrum, Viterbi tracking over spectral peaks; "
                     "no learned rate model",
           "protocol": "every window of every recording; rate has no trained parameters; confidence calibration "
                       "cross-fitted by subject",
           "reference": "spectral peak of WiRe-Breath gt_signal (chest belt) over the same window",
           "groups": out_groups, "comparison": comparison,
           "baseline": {"method": "classic spectral peak of the first principal component (amplitude)",
                        **comparison["classic (PCA peak, amplitude)"]},
           "synthetic_check": synth,
           "by_confidence": [{"lo": float(lo), "hi": float(hi), "mae": float(err[(conf >= lo) & (conf <= hi)].mean()),
                              "within2": float(np.mean(err[(conf >= lo) & (conf <= hi)] <= 2))}
                             for lo, hi in zip(edges[:-1], edges[1:])]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval_respiration.json").write_text(json.dumps(out, indent=1))
    for name, s in comparison.items():
        print(f"{name:32s} MAE={s['mae']:.2f} ±2={s['within2']:.0%} quiet={s['quiet_mae']:.2f} interf={s['interference_mae']:.2f}")
    for g, s in out_groups.items():
        print(f"{g:28s} n={s['n']:5d} MAE={s['mae']:.2f} ±2={s['within2']:.0%} gated={s['gated_mae']} cov={s['coverage']:.0%}")
    print("synthetic {true: [estimate, confidence]}:", synth)
    print("by confidence:", [(round(x['lo'], 2), round(x['mae'], 2)) for x in out["by_confidence"]])


if __name__ == "__main__":
    main()
