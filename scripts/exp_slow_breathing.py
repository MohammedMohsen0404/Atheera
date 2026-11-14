"""Slow-breathing test: does the estimator read the fundamental, or lock onto its 2nd harmonic?

Every WiRe-Breath recording is evaluated as recorded (x1.0) and slowed to 0.6x its rate (real CSI and chest-belt
reference re-timed identically, scripts/build_tracks.stretch_recording), which puts most true rates at 6-10 br/min.
Compares estimator band settings; the reference always uses the fixed wide band R.REF_BAND_HZ.

  python scripts/exp_slow_breathing.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from atheer.ingest import wirebreath as WB  # noqa: E402
from atheer.modules import respiration as R  # noqa: E402
from build_tracks import stretch_recording  # noqa: E402

SETTINGS = {  # name -> (estimator band Hz, lowest reported rate br/min)
    "band 0.10-0.6, report >=6": ((0.10, 0.6), 6.0),
    "band 0.07-0.6, report >=4.5": ((0.07, 0.6), 4.5),
    "band 0.05-0.6, report >=4": ((0.05, 0.6), 4.0),
}


def run(rec, win=20.0, hop=5.0):
    fs = rec.fs; n, h = int(win * fs), int(hop * fs)
    ends = list(range(n, rec.csi.shape[0] + 1, h))
    refs = [R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[e - n], rec.time[e - 1]) for e in ends]
    ch = R.channels(rec.csi, fs)
    wss = [R.window_spectrum(ch, slice(e - n, e), fs) for e in ends]
    path = R.track([R._peaks_only(w.fine) for w in wss], R.TRACK_LAM * 5.0 / hop)
    return [(R._parabolic(w.fine, j), ref, R.confidence(w, R._parabolic(w.fine, j))) for w, j, ref in zip(wss, path, refs)]


def main():
    files = WB.find_files(ROOT / "data" / "raw" / "wirebreath" / "extracted")
    res = defaultdict(list)
    for i, f in enumerate(files):
        base = WB.load(f)
        recs = {"x1.0": base, "x0.6": stretch_recording(base, 0.6)}
        for name, (band, lo) in SETTINGS.items():
            R.BAND_HZ = band
            R.FINE = np.arange(lo, 36.01, 0.1)
            for sname, rec in recs.items():
                for est, ref, conf in run(rec):
                    if np.isfinite(ref):
                        res[(name, sname)].append((est, ref, conf))
        if i % 50 == 0:
            print(f"{i}/{len(files)}", flush=True)
    print(f"{'setting':30s} {'data':5s}  MAE   ±2   2x-harmonic  confident&wrong(>3, conf>=0.45)  ref range")
    for (name, sname), rows in res.items():
        a = np.array(rows); e = np.abs(a[:, 0] - a[:, 1])
        harm = np.mean(np.abs(a[:, 0] / a[:, 1] - 2) < 0.2)
        cw = np.mean((e > 3) & (a[:, 2] >= 0.45))
        print(f"{name:30s} {sname:5s} {e.mean():5.2f} {np.mean(e <= 2):4.0%}  {harm:6.1%}       {cw:6.1%}"
              f"                      {np.quantile(a[:, 1], 0.05):.1f}-{np.quantile(a[:, 1], 0.95):.1f}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ---------------------------------------------------------------- sub-harmonic check variant
def run_subharmonic(rec, rho, win=20.0, hop=5.0, wide=(0.05, 0.6)):
    """Narrow band for peaks; a peak at f moves to f/2 when the wide-band spectrum holds >= rho of its power
    near f/2 (a genuine fundamental has nothing at half its rate; slow breathing seen through its harmonic does)."""
    fs = rec.fs; n, h = int(win * fs), int(hop * fs)
    ends = list(range(n, rec.csi.shape[0] + 1, h))
    refs = [R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[e - n], rec.time[e - 1]) for e in ends]
    R.BAND_HZ = (0.10, 0.6); R.FINE = np.arange(4.0, 36.01, 0.1)
    chn = R.channels(rec.csi, fs)
    R.BAND_HZ = wide
    chw = R.channels(rec.csi, fs)
    R.BAND_HZ = (0.10, 0.6)
    out = []
    scores, wsn = [], []
    for e in ends:
        wn = R.window_spectrum(chn, slice(e - n, e), fs)
        ww = R.window_spectrum(chw, slice(e - n, e), fs)
        sc = R._peaks_only(wn.fine)
        sc[R.FINE < 6.0] = 1e-6                       # narrow band cannot see below 6 itself
        for j in np.where(sc > 1e-3)[0]:
            half = R.FINE[j] / 2
            if half < 4.0:
                continue
            m_h = np.abs(R.FINE - half) <= 0.6; m_f = np.abs(R.FINE - R.FINE[j]) <= 0.6
            if ww.fine[m_h].max() >= rho * ww.fine[m_f].max():
                jh = np.where(m_h)[0][np.argmax(ww.fine[m_h])]
                sc[jh] = max(sc[jh], sc[j]); sc[j] = 1e-6
        scores.append(sc); wsn.append(wn)
    path = R.track(scores, R.TRACK_LAM * 5.0 / hop)
    for w, j, ref in zip(wsn, path, refs):
        bpm = R._parabolic(w.fine, j) if w.fine[j] > 0.2 else float(R.FINE[j])
        out.append((bpm, ref, R.confidence(w, bpm)))
    return out


def main_subharmonic():
    files = WB.find_files(ROOT / "data" / "raw" / "wirebreath" / "extracted")
    res = defaultdict(list)
    for i, f in enumerate(files):
        base = WB.load(f)
        for sname, rec in (("x1.0", base), ("x0.6", stretch_recording(base, 0.6))):
            for rho in (0.3, 0.5, 0.8):
                res[(f"subharmonic rho={rho}", sname)] += [r for r in run_subharmonic(rec, rho) if np.isfinite(r[1])]
        if i % 50 == 0:
            print(f"{i}/{len(files)}", flush=True)
    for (name, sname), rows in res.items():
        a = np.array(rows); e = np.abs(a[:, 0] - a[:, 1])
        harm = np.mean(np.abs(a[:, 0] / a[:, 1] - 2) < 0.2); half = np.mean(np.abs(a[:, 0] / a[:, 1] - 0.5) < 0.1)
        print(f"{name:24s} {sname:5s} MAE {e.mean():5.2f} ±2 {np.mean(e <= 2):4.0%}  2x {harm:5.1%}  0.5x {half:5.1%}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "subharmonic":
    main_subharmonic()


# ---------------------------------------------------------------- ambiguity flag variant
def run_both(rec, win=20.0, hop=5.0):
    fs = rec.fs; n, h = int(win * fs), int(hop * fs)
    ends = list(range(n, rec.csi.shape[0] + 1, h))
    refs = [R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[e - n], rec.time[e - 1]) for e in ends]
    outs = {}
    for name, band, lo in (("narrow", (0.10, 0.6), 6.0), ("wide", (0.05, 0.6), 4.0)):
        R.BAND_HZ = band; R.FINE = np.arange(lo, 36.01, 0.1)
        ch = R.channels(rec.csi, fs)
        wss = [R.window_spectrum(ch, slice(e - n, e), fs) for e in ends]
        path = R.track([R._peaks_only(w.fine) for w in wss], R.TRACK_LAM * 5.0 / hop)
        outs[name] = [(R._parabolic(w.fine, j), R.confidence(w, R._parabolic(w.fine, j))) for w, j in zip(wss, path)]
    R.BAND_HZ = (0.10, 0.6); R.FINE = np.arange(6.0, 36.01, 0.1)
    return [(nb, nc, wb, ref) for (nb, nc), (wb, _), ref in zip(outs["narrow"], outs["wide"], refs)]


def main_ambiguity():
    files = WB.find_files(ROOT / "data" / "raw" / "wirebreath" / "extracted")
    res = defaultdict(list)
    for i, f in enumerate(files):
        base = WB.load(f)
        for sname, rec in (("x1.0", base), ("x0.6", stretch_recording(base, 0.6))):
            res[sname] += [r for r in run_both(rec) if np.isfinite(r[3])]
        if i % 50 == 0:
            print(f"{i}/{len(files)}", flush=True)
    for sname, rows in res.items():
        a = np.array(rows); nb, nc, wb, ref = a.T
        e = np.abs(nb - ref)
        amb = np.abs(wb / nb - 0.5) < 0.12            # wide band sees half the narrow-band rate
        conf = np.where(amb, 0.0, nc)
        for label, c in (("current", nc), ("with ambiguity flag", conf)):
            ok = c >= 0.45
            print(f"{sname} {label:20s} MAE {e.mean():.2f}  coverage {ok.mean():5.1%}  confident&wrong(>3) {np.mean(ok & (e > 3)):5.1%}"
                  f"  gated MAE {e[ok].mean():.2f}")
        print(f"{sname} flagged ambiguous: {amb.mean():.1%}; of those truly wrong (>3): {np.mean(e[amb] > 3) if amb.any() else 0:.0%}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ambiguity":
    main_ambiguity()
