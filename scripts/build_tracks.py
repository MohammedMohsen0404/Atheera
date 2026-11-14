"""Precompute compact replay tracks (1 Hz features + 10 Hz waveform) from the public datasets.

  python scripts/build_tracks.py --wirebreath data/raw/wirebreath   # breathing tracks (real)
  python scripts/build_tracks.py --uthar data/raw/sensefi            # activity tracks (real)
  python scripts/build_tracks.py --synthetic                         # dev-only placeholders

Output: results/tracks/<id>.json  — small enough to commit / download from Kaggle.
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
from atheer.modules import respiration as R  # noqa: E402
from atheer.modules.anomaly import motion_index  # noqa: E402

OUT = ROOT / "results" / "tracks"
WAVE_HZ = 10


def _r(x, n=3):
    return [None if not np.isfinite(v) else round(float(v), n) for v in np.asarray(x, float)]


def save(track: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{track['id']}.json").write_text(json.dumps(track, separators=(",", ":")))
    print(f"  wrote {track['id']}: {len(track['t'])} s  ({track['source'][:70]})")


CSI_SUB = 16  # subcarriers kept for the dashboard's raw-channel view


def csi_snapshot(amp: np.ndarray, fs: float, n_sec: int | None = None) -> list[str]:
    """Raw CSI amplitude for display: 16 evenly spaced subcarriers at 10 Hz, z-scored per subcarrier,
    quantised to one byte, one hex string (10 rows x 16 bytes) per second."""
    amp = np.asarray(amp, float)
    if amp.ndim == 1:
        amp = amp[:, None]
    idx = np.linspace(0, amp.shape[1] - 1, CSI_SUB).round().astype(int)
    x = amp[:, idx]
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    step = max(1, int(round(fs / WAVE_HZ)))
    x = x[::step]
    q = np.clip((x + 2.5) / 5.0 * 255, 0, 255).astype(np.uint8)
    n_sec = n_sec or len(q) // WAVE_HZ
    out = []
    for s in range(n_sec):
        blk = q[s * WAVE_HZ:(s + 1) * WAVE_HZ]
        if len(blk) < WAVE_HZ:
            blk = np.vstack([blk, np.repeat(blk[-1:] if len(blk) else np.full((1, CSI_SUB), 128, np.uint8), WAVE_HZ - len(blk), 0)])
        out.append(blk.tobytes().hex())
    return out


# ------------------------------------------------------------------ breathing (WiRe-Breath)
def breath_features(rec, win_s=20.0):  # 20 s windows, as evaluated
    """Per-second BR / confidence / motion for one 30 s recording (estimates held before first full window)."""
    fs, amp = rec.fs, rec.amp
    n_sec = int(amp.shape[0] / fs)
    bpm, conf, mot, gt = [], [], [], []
    ests = {int(round(t)): e for t, e in R.sliding(rec.csi, fs, win_s=win_s, hop_s=1.0)}
    first = min(ests) if ests else None
    for s in range(n_sec):
        e = ests.get(max(s + 1, first)) if first else None
        bpm.append(e.bpm if e else np.nan)
        conf.append(e.confidence if e else 0.0)
        a = amp[int(s * fs):int((s + 1) * fs)]
        mot.append(motion_index(amp[max(0, int((s - 2) * fs)):int((s + 1) * fs)], fs, gain_norm=True)
                   if a.size else 0.0)
        t0 = rec.time[0] + max(0, s + 1 - win_s)
        gt.append(R.reference_rate(rec.gt_time, rec.gt_signal, t0, rec.time[0] + s + 1) if s + 1 >= win_s else np.nan)
    w, _ = R.breathing_waveform(amp, fs)
    step = max(1, int(round(fs / WAVE_HZ)))
    return dict(bpm=bpm, conf=conf, motion=mot, gt_bpm=gt, wave=w[::step][: n_sec * WAVE_HZ],
                csi=csi_snapshot(amp, fs, n_sec))


def build_wirebreath(root: Path, files_per_track: int = 10):
    from atheer.ingest import wirebreath as WB
    files = WB.find_files(root)
    print(f"WiRe-Breath: {len(files)} files")
    groups = defaultdict(list)
    for f in files:
        m = WB.parse_filename(f)
        groups[(m.get("subject", "S?"), m.get("posture", "Unknown"), m.get("environment", "Quiet"))].append(f)
    for (subj, posture, env), fl in sorted(groups.items()):
        fl = fl[:files_per_track]
        acc = defaultdict(list)
        srcs = []
        for f in fl:
            rec = WB.load(f)
            ft = breath_features(rec)
            for k, v in ft.items():
                acc[k].extend(list(v))
            srcs.append(f.name)
        n = len(acc["bpm"])
        save({"id": f"wb_{subj}_{posture}_{env}".lower(), "kind": "breath", "synthetic": False,
              "dataset": "WiRe-Breath (Ge et al., IEEE JBHI 2025, CC-BY-4.0)",
              "meta": {"subject": subj, "posture": posture, "environment": env},
              "source": "; ".join(srcs), "t": list(range(n)), "bpm": _r(acc["bpm"], 2),
              "conf": _r(acc["conf"]), "motion": _r(acc["motion"], 5), "gt_bpm": _r(acc["gt_bpm"], 2),
              "wave": _r(acc["wave"]), "wave_hz": WAVE_HZ, "csi": acc["csi"]})


def stretch_recording(rec, k: float):
    """Slow a real recording down by factor k (< 1 lowers every rate by k): CSI and chest-belt reference are
    re-timed identically, then CSI is resampled to the original rate. Used only to simulate slowing breathing
    (bradypnoea) from genuine CSI; tracks built this way are labelled as time-stretched."""
    from atheer.ingest import wirebreath as WB
    t0 = rec.time[0]
    ts = t0 + (rec.time - t0) / k
    tu = t0 + np.arange(int((ts[-1] - t0) * rec.fs)) / rec.fs
    return WB.BreathRecording(rec.path, WB._resample(rec.csi, ts, tu), tu, t0 + (rec.gt_time - t0) / k,
                              rec.gt_signal, rec.fs, {**rec.meta, "stretch": k})


def build_slowing(root: Path, subject: str = "S2", posture: str = "Normal", factors=(1.0, 0.92, 0.84, 0.78, 0.74, 0.7)):
    """FE1's 'breathing slows' event from real CSI: successive real recordings of the same sleeper, each slowed
    a little more. S2 breathes ~9.3 br/min, so x1.0 ... x0.7 ends near 6.5: below the 8 br/min sleep limit but
    inside the estimator's 6-36 br/min range (slower is out of range by design). The estimator has to find the
    slowdown in the CSI itself."""
    from atheer.ingest import wirebreath as WB
    files = [f for f in WB.find_files(root) if WB.parse_filename(f).get("subject") == subject
             and WB.parse_filename(f).get("posture") == posture and WB.parse_filename(f).get("environment") == "Quiet"]
    acc, srcs = defaultdict(list), []
    for f, k in zip(files, factors):
        ft = breath_features(stretch_recording(WB.load(f), k))
        for key, v in ft.items():
            acc[key].extend(list(v))
        srcs.append(f"{f.name} x{k}")
    n = len(acc["bpm"])
    save({"id": f"wbx_{subject}_slowing".lower(), "kind": "breath", "synthetic": False, "stretched": True,
          "dataset": "WiRe-Breath (Ge et al., IEEE JBHI 2025, CC-BY-4.0), real CSI time-stretched to simulate slowing breathing",
          "meta": {"subject": subject, "posture": posture, "environment": "Quiet", "stretch_factors": list(factors)},
          "source": "; ".join(srcs), "t": list(range(n)), "bpm": _r(acc["bpm"], 2),
          "conf": _r(acc["conf"]), "motion": _r(acc["motion"], 5), "gt_bpm": _r(acc["gt_bpm"], 2),
          "wave": _r(acc["wave"]), "wave_hz": WAVE_HZ, "csi": acc["csi"]})


# ------------------------------------------------------------------ activity (UT-HAR via SenseFi)
def build_uthar(root: Path, sec_per_sample: float = 2.0, per_class: int = 30):
    """Activity tracks from UT-HAR recordings the deployed model never saw (grouped test split), replayed as
    consecutive windows of the same recordings."""
    import torch
    from atheer.modules.activity import UT_HAR_CLASSES, ActivityRecognizer
    d = root / "UT_HAR"
    parts = [(np.load(open(d / "data" / f"X_{s}.csv", "rb")).reshape(-1, 250, 90).astype(np.float32),
              np.load(open(d / "label" / f"y_{s}.csv", "rb")).astype(int)) for s in ("train", "val", "test")]
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    grp = np.load(ROOT / "models" / "har_groups.npz")
    held = np.where(grp["gsplit"] == "test")[0]
    rec = ActivityRecognizer()
    fs_eff = 250 / sec_per_sample
    for c, name in enumerate(UT_HAR_CLASSES):
        cls = held[y[held] == c]
        order = cls[np.lexsort((grp["pos"][cls], grp["g"][cls]))][:per_class]  # recording by recording, in time order
        probs = []
        for i in range(0, len(order), 128):
            probs += [r.probs for r in rec.predict(X[order[i:i + 128]])]
        t, lab, pr, mot, true, csi = [], [], [], [], [], []
        for i, pb in zip(order, probs):
            m = motion_index(X[i], fs_eff)
            csi += csi_snapshot(X[i], fs_eff, int(sec_per_sample))
            for s_ in range(int(sec_per_sample)):
                t.append(len(t)); pr.append([round(pb[n], 4) for n in UT_HAR_CLASSES])
                lab.append(max(pb, key=pb.get)); mot.append(round(m, 5)); true.append(name)
        recs = sorted(set(int(g) for g in grp["g"][order]))
        save({"id": f"uthar_{name}", "kind": "activity", "synthetic": False,
              "dataset": f"UT-HAR (Yousefi et al. 2017) via SenseFi benchmark; model {rec.name}, held-out recordings",
              "meta": {"true_label": name, "classes": UT_HAR_CLASSES, "recordings": recs},
              "source": f"UT-HAR held-out recordings {recs[:4]}...", "t": t, "activity": lab,
              "probs": pr, "motion": mot, "true": true, "csi": csi})
    del torch


# ------------------------------------------------------------------ synthetic placeholders (dev only)
def build_synthetic(seed: int = 0):
    from atheer.modules.activity import UT_HAR_CLASSES
    rng = np.random.default_rng(seed)
    fs, dur = 50.0, 300

    def breath(id_, bpm_fn, noise=0.6, motion=0.002):
        t = np.arange(0, dur, 1 / fs)
        phase = 2 * np.pi * np.cumsum(bpm_fn(t) / 60) / fs
        sig = np.sin(phase)
        amp = 20 + np.outer(sig, rng.uniform(-1, 1, 40)) * 0.5 + rng.normal(0, noise, (t.size, 40))
        bpm, conf, mot = [], [], []
        for s in range(dur):
            a = amp[max(0, int((s - 19) * fs)):int((s + 1) * fs)]
            e = R.estimate(a, fs) if a.shape[0] >= 10 * fs else None
            bpm.append(e.bpm if e else np.nan); conf.append(e.confidence if e else 0.0)
            mot.append(motion * (1 + 0.2 * rng.standard_normal()))
        k0 = next(i for i, v in enumerate(bpm) if np.isfinite(v))  # loop-friendly: no warm-up gap
        bpm[:k0], conf[:k0] = [bpm[k0]] * k0, [conf[k0]] * k0
        w, _ = R.breathing_waveform(amp, fs)
        save({"id": id_, "kind": "breath", "synthetic": True, "dataset": "SYNTHETIC placeholder",
              "meta": {}, "source": "synthetic", "t": list(range(dur)), "bpm": _r(bpm, 2), "conf": _r(conf),
              "motion": _r(mot, 5), "gt_bpm": _r([bpm_fn(np.array([s]))[0] for s in range(dur)], 2),
              "wave": _r(w[::5][: dur * WAVE_HZ]), "wave_hz": WAVE_HZ, "csi": csi_snapshot(amp, fs, dur)})

    breath("syn_sleep_normal", lambda t: 13 + 0 * t)
    breath("syn_sleep_slowing", lambda t: np.maximum(13 - t / 25, 6))
    breath("syn_rest_noisy", lambda t: 16 + 0 * t, noise=3.0)
    for name in UT_HAR_CLASSES:
        n = 60
        p = np.full((n, 7), 0.02); p[:, UT_HAR_CLASSES.index(name)] = 0.88
        mot = {"fall": 0.08, "run": 0.05, "walk": 0.03}.get(name, 0.01)
        tt = np.arange(0, n, 0.02)
        burst = 1.0 + (6.0 * np.exp(-((tt - 3) ** 2) / 0.5) if name == "fall" else 0)
        sim = 20 + rng.normal(0, 1, (tt.size, 40)) * (mot * 60 * burst)[..., None] if np.ndim(burst) else             20 + rng.normal(0, 1, (tt.size, 40)) * mot * 60
        sim = sim + np.sin(2 * np.pi * np.outer(tt, rng.uniform(0.2, 2.5, 40)) * (1 + 3 * mot))
        save({"id": f"syn_{name}", "kind": "activity", "synthetic": True, "dataset": "SYNTHETIC placeholder",
              "meta": {"true_label": name}, "source": "synthetic", "t": list(range(n)), "activity": [name] * n,
              "probs": p.round(3).tolist(), "motion": list(mot * (1 + 0.2 * rng.standard_normal(n))), "true": [name] * n,
              "csi": csi_snapshot(sim, 50.0, n)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wirebreath", type=Path)
    ap.add_argument("--uthar", type=Path)
    ap.add_argument("--slowing", type=Path, help="only the time-stretched 'breathing slows' track")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        build_synthetic()
    if a.wirebreath:
        build_wirebreath(a.wirebreath)
        build_slowing(a.wirebreath)
    if a.slowing:
        build_slowing(a.slowing)
    if a.uthar:
        build_uthar(a.uthar)
