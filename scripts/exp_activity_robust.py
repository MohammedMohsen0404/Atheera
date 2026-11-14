"""Robustness + honest-split study for Atheer activity / fall detection on UT-HAR (SenseFi).

Stages (each one appends to results/exp_activity_robust.json; CPU-only friendly):
  python scripts/exp_activity_robust.py groups            # recover recordings from window overlap, leakage stats
  python scripts/exp_activity_robust.py ft                # fine-tune pretrained ResNet18 (per-window norm + aug)
  python scripts/exp_activity_robust.py p0                # SenseFi split: pretrained archs, ensemble, ft model
  python scripts/exp_activity_robust.py p1train --variant base|wnorm|robust   # grouped split, from scratch
  python scripts/exp_activity_robust.py p1                # grouped split evaluation (+ event-level fall rule)

Protocols
  P0 = SenseFi split (train / val / test files). Known leaky: windows overlap 60% and test windows share raw rows
       with train windows. Val is used for temperature + thresholds, X_test only for reporting.
  P1 = recordings recovered from row overlap, whole recordings assigned 70/10/20 (stratified by class) to
       train/val/test. Removes sample-level leakage; still same people + room (no subject IDs exist in UT-HAR).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from har_common import (FALL, PERTURBATIONS, ROOT, UT_HAR_CLASSES, ece, fit_temperature, grouped_split,
                        load_all, load_pretrained, metrics, model_logits, motion_features, nll, norm_minmax,
                        norm_window, perturb, recording_groups, sanitize, align_offset, softmax, train_model)

OUT = ROOT / "results" / "exp_activity_robust.json"
MODELS = ROOT / "models"
GROUPS = MODELS / "har_groups.npz"
torch.set_num_threads(8)


def save(key, value):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = value
    OUT.write_text(json.dumps(d, indent=1))


def pkey(kind, lvl):
    return "clean" if kind == "clean" else f"{kind}={lvl}"


def load_groups(X, y, split):
    if GROUPS.exists():
        z = np.load(GROUPS)
        return z["g"], z["pos"], z["gsplit"]
    g, pos, hist = recording_groups(X)
    gsplit = grouped_split(y, g, seed=0)
    np.savez(GROUPS, g=g, pos=pos, gsplit=gsplit)
    return g, pos, gsplit


# ------------------------------------------------------------------------------------------------ systems
class System:
    """name, arch(s), weight paths, normalisation -> logits on raw windows (ensembles average probabilities)."""

    def __init__(self, name, members, norm, pre=()):
        self.name, self.norm, self.pre = name, norm, pre
        self.members = [(arch, load_pretrained(arch, path=path, device="cpu")) for arch, path in members]
        self.T = 1.0

    def normalise(self, X, split_stats=None):
        if "sanitize" in self.pre:
            X = sanitize(X)
        if "align" in self.pre:
            X = align_offset(X, self.mu)
        if self.norm == "window":
            return norm_window(X)
        lo, hi = self.norm if split_stats is None else split_stats
        return norm_minmax(X, lo, hi)

    def logits(self, X, split_stats=None):
        Xn = self.normalise(X, split_stats)
        Ls = [model_logits(m, Xn, a) for a, m in self.members]
        if len(Ls) == 1:
            return Ls[0]
        P = np.mean([softmax(L) for L in Ls], 0)  # probability averaging; return log-probs as logits
        return np.log(np.clip(P, 1e-8, None))

    def probs(self, X, calibrated=True, split_stats=None):
        return softmax(self.logits(X, split_stats), self.T if calibrated else 1.0)


def f2(alert, isf):
    tp = (alert & isf).sum()
    fp = (alert & ~isf).sum()
    fn = (~alert & isf).sum()
    return 5 * tp / max(5 * tp + 4 * fn + fp, 1)


def pick_threshold(pf_val, isf_val):
    """Threshold on val maximising F2 (recall weighted 4x precision); ties -> higher threshold."""
    best = (-1, 0.5)
    for t in np.round(np.arange(0.95, 0.04, -0.01), 2):
        sc = f2(pf_val >= t, isf_val)
        if sc > best[0] + 1e-12:
            best = (sc, float(t))
    return best[1]


def evaluate(systems, Xva, yva, Xte, yte, perts=PERTURBATIONS, log=print):
    res, calib = {}, {}
    for s in systems:
        Lva = s.logits(Xva)
        s.T = fit_temperature(Lva, yva)
        Pva = softmax(Lva, s.T)
        s.thr = pick_threshold(Pva[:, FALL], yva == FALL)
        Lte = s.logits(Xte)
        P0, P1 = softmax(Lte), softmax(Lte, s.T)
        calib[s.name] = {"temperature": round(s.T, 3), "fall_threshold_from_val": s.thr,
                         "test_ece_before": round(ece(P0, yte), 4), "test_ece_after": round(ece(P1, yte), 4),
                         "test_nll_before": round(nll(P0, yte), 4), "test_nll_after": round(nll(P1, yte), 4),
                         "val_ece_before": round(ece(softmax(Lva), yva), 4), "val_ece_after": round(ece(Pva, yva), 4)}
        log(f"{s.name}: T={s.T:.2f} thr={s.thr} ECE {calib[s.name]['test_ece_before']} -> "
            f"{calib[s.name]['test_ece_after']}")
        res[s.name] = {}
    for i, (kind, lvl) in enumerate(perts):
        Xp = perturb(Xte, kind, lvl, seed=1000 + i)
        for s in systems:
            L = s.logits(Xp)
            r = metrics(softmax(L), yte, 0.6)  # raw probs, fixed 0.6 threshold (current Atheer rule)
            rc = metrics(softmax(L, s.T), yte, s.thr)  # calibrated probs, threshold chosen on clean val
            r.update({"cal_fall_recall": rc["fall_recall"], "cal_fall_false_alarms": rc["fall_false_alarms"],
                      "ece_cal": round(ece(softmax(L, s.T), yte), 4)})
            res[s.name][pkey(kind, lvl)] = r
        log(f"  {pkey(kind, lvl):28s} " + "  ".join(
            f"{s.name[:14]}={res[s.name][pkey(kind, lvl)]['acc']:.3f}" for s in systems))
    return res, calib


def norm_mismatch(systems, Xte, yte, lo_hi_train):
    """Test-time normalisation variants for the min-max (SenseFi) models."""
    out = {}
    lo, hi = lo_hi_train
    rng = (hi - lo)
    variants = {
        "split_minmax_test (SenseFi protocol)": (float(Xte.min()), float(Xte.max())),
        "train_minmax (fixed, deploy)": (lo, hi),
        "minmax_range_+20pct (other env stats)": (lo - 0.1 * rng, hi + 0.1 * rng),
        "minmax_range_-20pct (other env stats)": (lo + 0.1 * rng, hi - 0.1 * rng),
        "minmax_shift_+3dB (other env stats)": (lo + 3, hi + 3),
    }
    for s in systems:
        if s.norm == "window" or "+ft" in s.name:
            continue
        out[s.name] = {}
        for k, st in variants.items():
            out[s.name][k] = metrics(s.probs(Xte, calibrated=False, split_stats=st), yte, 0.6)
        # per-window min-max (what a streaming system without global stats would do)
        Xn = (Xte - Xte.min((1, 2), keepdims=True)) / (np.ptp(Xte, axis=(1, 2), keepdims=True) + 1e-6)
        Ls = [model_logits(m, Xn, a) for a, m in s.members]
        P = np.mean([softmax(L) for L in Ls], 0)
        out[s.name]["per_window_minmax"] = metrics(P, yte, 0.6)
    return out


# ----------------------------------------------------------------------------------------- fall rules
def fall_rule_study(system, Xva, yva, Xte, yte, perts=(("clean", 0), ("noise_snr_db", 5), ("time_shift", 50),
                                                       ("packet_loss", 0.3), ("agc_per_antenna_db", 4))):
    """Window-level: fall prob only vs fall prob AND motion burst AND post-event stillness.
    All thresholds chosen on val (max F2), reported on test."""
    Pva = system.probs(Xva)
    fva = motion_features(Xva)
    isf = yva == FALL
    # candidate rule grid
    best = None
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        for bq in (0, 0.02, 0.05, 0.1, 0.2):
            for tq in (1.0, 0.98, 0.95, 0.9, 0.8):
                b = np.quantile(fva["burst"][isf], bq)
                ta = np.quantile(fva["tail"][isf], tq)
                al = (Pva[:, FALL] >= t) & (fva["burst"] >= b) & (fva["tail"] <= ta)
                sc = f2(al, isf)
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, float(al[isf].mean()), int(al[~isf].sum()), float(t), float(b), float(ta))
    best = (best[2], best[1], best[3], best[4], best[5])
    _, _, t, b, ta = best
    t_only = system.thr
    out = {"rule": f"p_fall>={t} & burst>={b:.3f} & tail<={ta:.3f}", "prob_only_threshold": t_only,
           "val_rule": {"recall": round(best[1], 4), "false_alarms": best[0]},
           "val_prob_only": {"recall": round(float((Pva[isf, FALL] >= t_only).mean()), 4),
                             "false_alarms": int((Pva[~isf, FALL] >= t_only).sum())},
           "motion_feature_medians_val": {
               UT_HAR_CLASSES[c]: {k: round(float(np.median(v[yva == c])), 4) for k, v in fva.items()}
               for c in range(7)}}
    for i, (kind, lvl) in enumerate(perts):
        Xp = perturb(Xte, kind, lvl, seed=5000 + i)
        P = system.probs(Xp)
        f = motion_features(Xp)
        isf_t = yte == FALL
        a1 = P[:, FALL] >= t_only
        a2 = (P[:, FALL] >= t) & (f["burst"] >= b) & (f["tail"] <= ta)
        out[pkey(kind, lvl)] = {
            "prob_only": {"recall": round(float(a1[isf_t].mean()), 4), "false_alarms": int(a1[~isf_t].sum())},
            "prob+burst+stillness": {"recall": round(float(a2[isf_t].mean()), 4),
                                     "false_alarms": int(a2[~isf_t].sum())}}
    return out


def event_level(system, X, y, g, pos, thr, perts=(("clean", 0), ("noise_snr_db", 5), ("packet_loss", 0.3))):
    """Recording-level fall alarms on the grouped test: windows of each recording in time order (stride 100 rows);
    alarm if >= k consecutive windows have p_fall >= thr. Counts fall recordings detected and non-fall
    recordings that raised any alarm."""
    out = {}
    for i, (kind, lvl) in enumerate(perts):
        P = system.probs(perturb(X, kind, lvl, seed=7000 + i))[:, FALL]
        r = {}
        for k in (1, 2):
            det = fa = nf = nn = 0
            for gid in np.unique(g):
                m = np.where(g == gid)[0]
                m = m[np.argsort(pos[m])]
                hit = P[m] >= thr
                run = best = 0
                for h in hit:
                    run = run + 1 if h else 0
                    best = max(best, run)
                alarm = best >= min(k, len(m))
                if y[m[0]] == FALL:
                    nf += 1
                    det += alarm
                else:
                    nn += 1
                    fa += alarm
            r[f"k={k}"] = {"fall_recordings_detected": f"{det}/{nf}", "nonfall_recordings_with_alarm": f"{fa}/{nn}"}
        out[pkey(kind, lvl)] = r
    return out


# ------------------------------------------------------------------------------------------------ stages
def stage_groups(X, y, split):
    t0 = time.time()
    g, pos, hist = recording_groups(X)
    gsplit = grouped_split(y, g, seed=0)
    np.savez(GROUPS, g=g, pos=pos, gsplit=gsplit)
    te = np.where(split == "test")[0]
    tr_groups = set(g[split == "train"])
    va_groups = set(g[split == "val"])
    leak = {
        "n_windows": int(len(X)), "n_recordings_recovered": int(len(np.unique(g))),
        "recordings_per_class": {UT_HAR_CLASSES[c]: int(len(np.unique(g[y == c]))) for c in range(7)},
        "recording_size_windows": {"min": int(np.bincount(g).min()), "median": float(np.median(np.bincount(g))),
                                   "max": int(np.bincount(g).max())},
        "label_pure_recordings": int(sum(len(np.unique(y[g == k])) == 1 for k in np.unique(g))),
        "pair_overlap_rows_hist (top)": {str(k): v for k, v in hist.items()},
        "senseFi_test_windows_whose_recording_is_also_in_train": int(sum(g[i] in tr_groups for i in te)),
        "senseFi_val_windows_whose_recording_is_also_in_train": int(sum(g[i] in tr_groups for i in np.where(split == "val")[0])),
        "senseFi_test_windows_whose_recording_is_also_in_val": int(sum(g[i] in va_groups for i in te)),
        "grouped_split_sizes": {s: int((gsplit == s).sum()) for s in ("train", "val", "test")},
        "grouped_split_recordings": {s: int(len(np.unique(g[gsplit == s]))) for s in ("train", "val", "test")},
        "grouped_split_fall_recordings": {s: int(len(np.unique(g[(gsplit == s) & (y == FALL)])))
                                          for s in ("train", "val", "test")},
        "seconds": round(time.time() - t0, 1)}
    print(json.dumps(leak, indent=1))
    save("leakage_and_grouping", leak)


def stage_ft(X, y, split):
    tr, va = split == "train", split == "val"
    t0 = time.time()
    model, ep = train_model("ResNet18", X[tr], y[tr], X[va], y[va], norm_window, epochs=8, lr=3e-4, aug=True,
                            init=ROOT / "models" / "sensefi" / "UT-HAR" / "UT_HAR_ResNet18.pt", seed=0)
    path = MODELS / "har_resnet18_ft_wnorm_aug.pt"
    torch.save(model.state_dict(), path)
    save("ft_training", {"model": str(path.relative_to(ROOT)), "init": "SenseFi pretrained ResNet18",
                         "data": "SenseFi X_train (val for epoch selection)", "norm": "per-window standardisation",
                         "epochs": 8, "best_epoch": ep, "lr": 3e-4, "augmentation": True,
                         "minutes": round((time.time() - t0) / 60, 1)})


def stage_p0(X, y, split):
    tr, va, te = split == "train", split == "val", split == "test"
    lohi = (float(X[tr].min()), float(X[tr].max()))
    W = ROOT / "models" / "sensefi" / "UT-HAR"
    systems = [System(f"pre_{a}", [(a, W / f"UT_HAR_{a}.pt")], lohi) for a in ("ResNet18", "CNN+GRU", "BiLSTM")]
    systems.append(System("pre_ensemble3", [(a, W / f"UT_HAR_{a}.pt") for a in ("ResNet18", "CNN+GRU", "BiLSTM")],
                          lohi))
    ft = MODELS / "har_resnet18_ft_wnorm_aug.pt"
    if ft.exists():
        systems.append(System("ft_ResNet18_wnorm_aug", [("ResNet18", ft)], "window"))
        systems.append(System("ens_pre_ResNet18+ft", [("ResNet18", W / "UT_HAR_ResNet18.pt")], lohi))
    # ensemble of pretrained (min-max) + fine-tuned (window norm) needs mixed normalisation
    if ft.exists():
        pre, fts = systems[0], systems[4]

        def mixed_logits(Xr, split_stats=None, _p=pre, _f=fts):
            P = (softmax(_p.logits(Xr)) + softmax(_f.logits(Xr))) / 2
            return np.log(np.clip(P, 1e-8, None))
        systems[5].logits = mixed_logits
    res, calib = evaluate(systems, X[va], y[va], X[te], y[te])
    nm = norm_mismatch(systems, X[te], y[te], lohi)
    best = systems[4] if ft.exists() else systems[0]
    fr = {s.name: fall_rule_study(s, X[va], y[va], X[te], y[te]) for s in (systems[0], best)}
    save("p0_senseFi_split", {
        "test": "SenseFi X_test (500 windows; 45 falls)", "val": "SenseFi X_val (496; temperature + thresholds)",
        "baseline_norm": f"fixed train-split min-max {lohi}", "perturbed": res, "calibration": calib,
        "normalisation_mismatch": nm, "fall_rule": fr})


def stage_p0b(X, y, split):
    """Test-time fixes without retraining (+ the fine-tuned model with the same fixes)."""
    tr, va, te = split == "train", split == "val", split == "test"
    lohi = (float(X[tr].min()), float(X[tr].max()))
    mu = float(X[tr].mean())
    W = ROOT / "models" / "sensefi" / "UT-HAR"
    ft = MODELS / "har_resnet18_ft_wnorm_aug.pt"
    R = [("ResNet18", W / "UT_HAR_ResNet18.pt")]
    systems = [System("pre_ResNet18", R, lohi),
               System("pre_ResNet18+sanitize", R, lohi, ("sanitize",)),
               System("pre_ResNet18+sanitize+align", R, lohi, ("sanitize", "align")),
               System("pre_ensemble3+sanitize+align", [(a, W / f"UT_HAR_{a}.pt") for a in ("ResNet18", "CNN+GRU", "BiLSTM")],
                      lohi, ("sanitize", "align")),
               System("ft_ResNet18_wnorm_aug+sanitize", [("ResNet18", ft)], "window", ("sanitize",))]
    for s in systems:
        s.mu = mu
    res, calib = evaluate(systems, X[va], y[va], X[te], y[te])
    save("p0_testtime_fixes", {"train_mean_db": mu, "perturbed": res, "calibration": calib})


def stage_p1train(X, y, split, variant):
    g, pos, gs = load_groups(X, y, split)
    tr, va = gs == "train", gs == "val"
    lohi = (float(X[tr].min()), float(X[tr].max()))
    normfn = norm_window if variant in ("wnorm", "robust") else (lambda Z: norm_minmax(Z, *lohi))
    t0 = time.time()
    model, ep = train_model("ResNet18", X[tr], y[tr], X[va], y[va], normfn, epochs=25, lr=1e-3,
                            aug=(variant == "robust"), seed=0)
    path = MODELS / f"har_resnet18_grp_{variant}.pt"
    torch.save(model.state_dict(), path)
    d = json.loads(OUT.read_text()).get("p1_training", {}) if OUT.exists() else {}
    d[variant] = {"model": str(path.relative_to(ROOT)), "init": "scratch", "epochs": 25, "best_epoch": ep,
                  "lr": 1e-3, "norm": "window" if variant != "base" else f"train min-max {lohi}",
                  "augmentation": variant == "robust", "minutes": round((time.time() - t0) / 60, 1)}
    save("p1_training", d)


def stage_p1(X, y, split):
    g, pos, gs = load_groups(X, y, split)
    tr, va, te = gs == "train", gs == "val", gs == "test"
    lohi = (float(X[tr].min()), float(X[tr].max()))
    systems = []
    for v in ("base", "wnorm", "robust"):
        p = MODELS / f"har_resnet18_grp_{v}.pt"
        if p.exists():
            systems.append(System(f"grp_{v}", [("ResNet18", p)], "window" if v != "base" else lohi))
            if v == "base":
                systems.append(System("grp_base+sanitize+align", [("ResNet18", p)], lohi, ("sanitize", "align")))
            else:
                systems.append(System(f"grp_{v}+sanitize", [("ResNet18", p)], "window", ("sanitize",)))
    for s in systems:
        s.mu = float(X[tr].mean())
    res, calib = evaluate(systems, X[va], y[va], X[te], y[te])
    nm = norm_mismatch(systems, X[te], y[te], lohi)
    fr, ev = {}, {}
    for s in systems:
        fr[s.name] = fall_rule_study(s, X[va], y[va], X[te], y[te])
        ev[s.name] = event_level(s, X[te], y[te], g[te], pos[te], s.thr)
    save("p1_grouped_split", {
        "test": f"{int(te.sum())} windows from {len(np.unique(g[te]))} held-out recordings "
                f"({int((y[te] == FALL).sum())} fall windows, {len(np.unique(g[te & (y == FALL)]))} fall recordings)",
        "perturbed": res, "calibration": calib, "normalisation_mismatch": nm, "fall_rule": fr,
        "event_level": ev})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["groups", "ft", "p0", "p0b", "p1train", "p1"])
    ap.add_argument("--variant", default="robust", choices=["base", "wnorm", "robust"])
    a = ap.parse_args()
    X, y, split = load_all()
    t0 = time.time()
    if a.stage == "groups":
        stage_groups(X, y, split)
    elif a.stage == "ft":
        stage_ft(X, y, split)
    elif a.stage == "p0":
        stage_p0(X, y, split)
    elif a.stage == "p0b":
        stage_p0b(X, y, split)
    elif a.stage == "p1train":
        stage_p1train(X, y, split, a.variant)
    else:
        stage_p1(X, y, split)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
