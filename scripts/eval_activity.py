"""Evaluate activity recognition and fall detection on UT-HAR.

Two protocols:
  1. Reproduction (``all_archs``): SenseFi's pretrained models, unchanged, on SenseFi's own split (X_val + X_test,
     per-split min-max) - shows the published models behave as reported (ResNet18 98.3%).
     That split LEAKS: UT-HAR windows are 250-row slices (stride 100) of 556 continuous recordings, and 499 of
     the 500 test windows come from recordings that also supply training windows.
  2. Headline (``accuracy``, ``per_class``, fall numbers): the deployed robust model on whole recordings it never
     saw (grouped split from scripts/exp_activity_robust.py, models/har_groups.npz). The fall rule (probability
     threshold + consecutive windows) is chosen on the grouped validation split and only then applied to test.

Writes results/eval_activity.json and results/eval_anomaly.json.
  python scripts/eval_activity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.modules.activity import UT_HAR_CLASSES, ActivityRecognizer  # noqa: E402

FALL = UT_HAR_CLASSES.index("fall")


def load_split(d: Path, name: str):
    X = np.load(open(d / "data" / f"X_{name}.csv", "rb")).reshape(-1, 250, 90).astype(np.float32)
    y = np.load(open(d / "label" / f"y_{name}.csv", "rb")).astype(int)
    return X, y


def probs(rec: ActivityRecognizer, X: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, len(X), 128):
        out += [[r.probs[c] for c in UT_HAR_CLASSES] for r in rec.predict(X[i:i + 128])]
    return np.array(out)


def confusion(y, pred):
    cm = np.zeros((7, 7), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def fall_events(pf: np.ndarray, g: np.ndarray, pos: np.ndarray, thr: float, k: int) -> dict[int, bool]:
    """Recording id -> alarm raised (k consecutive windows of that recording at or above thr)."""
    out = {}
    for rid in np.unique(g):
        m = g == rid
        seq = pf[m][np.argsort(pos[m])] >= thr
        run = best = 0
        for v in seq:
            run = run + 1 if v else 0
            best = max(best, run)
        out[int(rid)] = best >= k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "raw" / "sensefi")
    ap.add_argument("--archs", nargs="+", default=["CNN+GRU", "ResNet18", "BiLSTM"])
    a = ap.parse_args()
    d = a.root / "UT_HAR"

    # ---- 1. reproduction on SenseFi's split
    splits = [load_split(d, s) for s in ("val", "test")]
    all_archs = {}
    for arch in a.archs:
        try:
            P = np.concatenate([probs(ActivityRecognizer(arch, norm=(float(X.min()), float(X.max())), robust=False), X)
                                for X, _ in splits])
        except FileNotFoundError as e:
            print(f"skip {arch}: {e}")
            continue
        Y = np.concatenate([y for _, y in splits])
        all_archs[arch] = float((P.argmax(1) == Y).mean())
        print(f"SenseFi split  {arch:9s} accuracy {all_archs[arch]:.4f}")

    # ---- 2. deployed model on held-out recordings
    parts = {s: load_split(d, s) for s in ("train", "val", "test")}
    X = np.concatenate([parts[s][0] for s in parts]); y = np.concatenate([parts[s][1] for s in parts])
    grp = np.load(ROOT / "models" / "har_groups.npz")
    g, pos, gsplit = grp["g"], grp["pos"], grp["gsplit"]
    rec = ActivityRecognizer()
    if not rec.robust:
        sys.exit("robust model weights missing: models/har_resnet18_grp_robust.pt (scripts/exp_activity_robust.py)")
    res = {}
    for s in ("val", "test"):
        m = gsplit == s
        res[s] = dict(P=probs(rec, X[m]), y=y[m], g=g[m], pos=pos[m])
    # fall rule chosen on validation recordings: smallest number of false recordings with all falls caught,
    # preferring the shortest persistence (fastest alarm)
    v = res["val"]
    best = None
    for k in (1, 2, 3):
        for thr in np.arange(0.2, 0.95, 0.05):
            ev = fall_events(v["P"][:, FALL], v["g"], v["pos"], thr, k)
            lab = {int(r): int(np.bincount(v["y"][v["g"] == r]).argmax()) for r in np.unique(v["g"])}
            caught = np.mean([ev[r] for r in ev if lab[r] == FALL])
            false = sum(ev[r] for r in ev if lab[r] != FALL)
            key = (-caught, false, k, -thr)
            if best is None or key < best[0]:
                best = (key, float(round(thr, 2)), k)
    _, thr, k = best
    t = res["test"]
    pred = t["P"].argmax(1); cm = confusion(t["y"], pred)
    per = {UT_HAR_CLASSES[c]: {"recall": float(cm[c, c] / max(cm[c].sum(), 1)),
                               "precision": float(cm[c, c] / max(cm[:, c].sum(), 1))} for c in range(7)}
    acc = float((pred == t["y"]).mean())
    print(f"held-out recordings: accuracy {acc:.4f} on {len(pred)} windows / {len(np.unique(t['g']))} recordings")
    exp = json.loads((ROOT / "results" / "exp_activity_robust.json").read_text()) if (ROOT / "results" / "exp_activity_robust.json").exists() else {}
    out = {"model": rec.name, "n": int(len(pred)), "n_recordings": int(len(np.unique(t["g"]))), "accuracy": acc,
           "classes": UT_HAR_CLASSES, "per_class": per, "confusion": cm.tolist(),
           "protocol": "whole UT-HAR recordings held out (70/10/20 by class); same people and room in every split",
           "all_archs": all_archs,
           "all_archs_protocol": "SenseFi split (X_val + X_test), pretrained weights unchanged - leaky, reproduction only",
           "leakage_note": "499 of SenseFi's 500 test windows come from recordings that also supply training windows",
           "sensefi_recipe_on_heldout_recordings": exp.get("summary", {}).get("honest_accuracy_resnet18_senseFi_recipe")}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval_activity.json").write_text(json.dumps(out, indent=1))

    from sklearn.metrics import roc_auc_score
    is_fall = (t["y"] == FALL).astype(int); pf = t["P"][:, FALL]
    alert = pf >= thr
    ev = fall_events(pf, t["g"], t["pos"], thr, k)
    lab = {int(r): int(np.bincount(t["y"][t["g"] == r]).argmax()) for r in np.unique(t["g"])}
    fall_recs = [r for r in ev if lab[r] == FALL]; other_recs = [r for r in ev if lab[r] != FALL]
    an = {"model": rec.name, "roc_auc": float(roc_auc_score(is_fall, pf)), "threshold": thr, "consecutive_windows": k,
          "recall": float(alert[is_fall == 1].mean()), "false_per_100": float(100 * alert[is_fall == 0].mean()),
          "n_fall": int(is_fall.sum()), "n_other": int((1 - is_fall).sum()),
          "recordings": {"falls": len(fall_recs), "falls_caught": int(sum(ev[r] for r in fall_recs)),
                         "other": len(other_recs), "false_alarms": int(sum(ev[r] for r in other_recs))},
          "protocol": "held-out recordings; threshold and persistence chosen on the validation recordings"}
    (ROOT / "results" / "eval_anomaly.json").write_text(json.dumps(an, indent=1))
    print("fall detection:", json.dumps(an))


if __name__ == "__main__":
    main()
