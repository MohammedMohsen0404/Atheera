"""Breathing-rate representations / fusion / rate estimators on WiRe-Breath, 20 s windows (hop 5 s).

Same protocol as scripts/eval_respiration.py / exp_cscr.py: the reference rate is the zero-padded Hann spectral
peak (6-36 br/min) of the band-passed chest belt over the same window; metric = MAE br/min and share within ±2.
Works from the 20 Hz complex cache written by scripts/exp_cscr.py (data/cache/wirebreath_complex.npy).

Data: own cache data/cache/repr_csi20.npy built from the raw .mat files (scripts/breath_repr_sanitise.py):
only positions 0..52 of the 234 reported subcarriers carry a channel; per-packet phase (CFO offset + timing slope)
is removed BEFORE resampling; |H| is resampled directly (never interpolate raw complex: its phase is random).
Representations (each turned into a set of real channels per window):
  amp234   |H| of all 234 columns (old pipeline input)      amp    |H| of the ~52 valid subcarriers
  san      sanitised complex H, projected per window per subcarrier onto its max-variance axis (FarSense-like)
  cscr1    per-packet ratio Hs(m)/Hs(ref), 1 reference (highest mean/std |H|), projected likewise
  cscr4    ratios to 4 references pooled       conj1  Hs(m) conj(Hs(ref)) / mean|Hs(ref)|^2
  rawc     unsanitised complex (control)       amp+cscr1, amp+san: channel sets pooled
Combiners (channels -> waveform weights or spectrum):
  pca      top-30 by band-power ratio, z-score, first principal component   (current classic pipeline)
  sum      top-30 by band-power ratio, summed normalised spectra             (current control)
  coh      top-30 by per-channel peak SNR, sign-aligned to the best channel, SNR-weighted coherent sum
  rfi      RespirFi: cosine-similarity matrix, leading eigenvector splits two sign groups, negate one,
           average the high-|loading| channels, Savitzky-Golay smoothing
Rate estimators (from the waveform, or the summed spectrum for `sum`):
  spec     spectral peak                 acf   highest autocorrelation peak at lag 60/36..60/6 s
  pint     median peak interval          hs    harmonic sum S(f)+S(2f) on the 0.1-1.2 Hz version
  hs5      S(f)+0.5 S(2f)                sxa   spectrum x autocorrelation (both must agree)
None of them uses a prior over typical breathing rates (every rate in 6-36 br/min is treated alike).

  python scripts/exp_breath_repr.py            # real data (results cached in data/cache/repr_rows.pkl)
  python scripts/exp_breath_repr.py --synth    # synthetic check only
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.modules import respiration as R  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from breath_repr_sanitise import sanitise  # noqa: E402

RAW_ROOT = ROOT / "data" / "raw" / "wirebreath" / "extracted"
CACHE = ROOT / "data" / "cache" / "repr_csi20.npy"   # own cache: correct amplitude + sanitised complex at 20 Hz
OUT = ROOT / "data" / "cache" / "repr_rows.pkl"
FS = 20.0
BAND = (0.1, 0.6)
WBAND = (0.1, 1.2)
LO, HI = 6.0, 36.0
WIN, HOP = 20.0, 5.0
TOPK = 30
PAD = 16


# ----------------------------------------------------------------------------------------------- preprocessing
def prep(x: np.ndarray, band) -> np.ndarray:
    """Hampel -> detrend -> band-pass over a whole recording; complex input is processed as re/im."""
    if np.iscomplexobj(x):
        y = prep(np.concatenate([x.real, x.imag], 1), band)
        m = x.shape[1]
        return y[:, :m] + 1j * y[:, m:]
    x = R.hampel(np.asarray(x, float))
    x = signal.detrend(x, axis=0)
    return R.bandpass(x, FS, band)


def _to_grid(t, x, tu, fs_in):
    """Uniform-rate resample of per-packet values x (T, M) (real or already-sanitised complex): linear
    interpolation onto a uniform grid at the packet rate, 4 Hz low-pass, then sampling at the 20 Hz grid."""
    tg = t[0] + np.arange(int((t[-1] - t[0]) * fs_in)) / fs_in
    def one(v):
        g = np.stack([np.interp(tg, t, v[:, j]) for j in range(v.shape[1])], 1)
        g = signal.sosfiltfilt(signal.butter(4, 4.0, fs=fs_in, output="sos"), g, axis=0)
        return np.stack([np.interp(tu, tg, g[:, j]) for j in range(g.shape[1])], 1)
    if np.iscomplexobj(x):
        return one(x.real) + 1j * one(x.imag)
    return one(x)


def ref_subcarriers(c: np.ndarray, n: int) -> np.ndarray:
    a = np.abs(c)
    return np.argsort(a.mean(0) / (a.std(0) + 1e-9))[::-1][:n]


def build_one(f):
    from atheer.ingest import wirebreath as WB
    raw = WB._load_raw(f)
    c = np.asarray(raw["csi_complex"])
    t = np.asarray(raw["time"], float).ravel()
    if c.shape[0] != t.size:
        c = c.T
    meta = WB.parse_filename(f)
    meta = {k: meta.get(k) for k in ("subject", "posture", "environment", "room", "file")}
    return build_from_arrays(t, c, np.asarray(raw["gt_time"], float).ravel(),
                             np.asarray(raw["gt_signal"], float).ravel(), meta)


def build_from_arrays(t, c, gt_t, gt_s, meta):
    o = np.argsort(t); t, c = t[o], c[o]
    keep = np.concatenate([[True], np.diff(t) > 0]); t, c = t[keep], c[keep].astype(complex)
    fs_in = (t.size - 1) / (t[-1] - t[0])
    tu = t[0] + np.arange(int((t[-1] - t[0]) * FS)) / FS
    hs, pos, _, _ = sanitise(c)
    r1 = ref_subcarriers(hs, 1)[0]
    ratio = np.delete(hs / hs[:, [r1]], r1, axis=1)            # per-packet CSCR, before any filtering
    out = dict(amp234=_to_grid(t, np.abs(c), tu, fs_in).astype(np.float32),
               san=_to_grid(t, hs, tu, fs_in).astype(np.complex64),
               cscr1=_to_grid(t, ratio, tu, fs_in).astype(np.complex64),
               rawc=_to_grid(t, c[:, pos], tu, fs_in).astype(np.complex64),   # control: NOT sanitised
               pos=pos, gt=np.interp(tu, gt_t, gt_s).astype(np.float32), meta=meta)
    return out


def build_cache(jobs=8):
    from joblib import Parallel, delayed
    from atheer.ingest import wirebreath as WB
    files = WB.find_files(RAW_ROOT)
    recs = Parallel(n_jobs=jobs, verbose=2)(delayed(build_one)(f) for f in files)
    np.save(CACHE, np.array(recs, dtype=object), allow_pickle=True)


def representations(r: dict) -> dict:
    """Cached recording -> {name: raw (unfiltered) array at 20 Hz, real or complex}."""
    hs = r["san"].astype(complex)
    pos = r["pos"]
    out = {"amp234": r["amp234"].astype(float),                     # current pipeline input (all 234 columns)
           "amp": r["amp234"][:, pos].astype(float),                # valid subcarriers only
           "san": hs,                                               # sanitised complex
           "cscr1": r["cscr1"].astype(complex),                     # per-packet ratio to 1 reference
           "rawc": r["rawc"].astype(complex)}                       # unsanitised complex (control)
    cols = []
    for ref in ref_subcarriers(hs, 4):
        cols.append(np.delete(hs / hs[:, [ref]], ref, axis=1))
    out["cscr4"] = np.concatenate(cols, 1)
    r1 = ref_subcarriers(hs, 1)[0]
    out["conj1"] = np.delete(hs * np.conj(hs[:, [r1]]), r1, axis=1) / np.mean(np.abs(hs[:, r1]) ** 2)
    return out


def project(zn: np.ndarray, zw: np.ndarray | None = None):
    """Per-channel projection of complex band-passed windows onto the axis of maximum (breathing-band) variance.
    The same angle is applied to the wide-band version so both stay comparable."""
    re, im = zn.real, zn.imag
    re = re - re.mean(0); im = im - im.mean(0)
    th = 0.5 * np.arctan2(2 * (re * im).mean(0), (re ** 2).mean(0) - (im ** 2).mean(0))
    pn = re * np.cos(th) + im * np.sin(th)
    if zw is None:
        return pn, None
    pw = (zw.real - zw.real.mean(0)) * np.cos(th) + (zw.imag - zw.imag.mean(0)) * np.sin(th)
    return pn, pw


# --------------------------------------------------------------------------------------------------- spectra
def spectrum(x: np.ndarray):
    x = x - x.mean(0)
    w = np.hanning(len(x))
    x = x * (w[:, None] if x.ndim == 2 else w)
    n = int(2 ** np.ceil(np.log2(len(x) * PAD)))
    return np.fft.rfftfreq(n, 1 / FS) * 60.0, np.abs(np.fft.rfft(x, n, axis=0)) ** 2


def interp_peak(f, S, idx):
    i = int(idx)
    if 0 < i < len(S) - 1:
        a, b, c = S[i - 1], S[i], S[i + 1]
        d = a - 2 * b + c
        delta = 0.5 * (a - c) / d if d != 0 else 0.0
        return float(f[i] + np.clip(delta, -0.5, 0.5) * (f[1] - f[0]))
    return float(f[i])


def peak_in_band(f, S):
    inb = np.where((f >= LO) & (f <= HI))[0]
    return interp_peak(f, S, inb[np.argmax(S[inb])])


def S_at(f, S, g):
    """Spectrum value at arbitrary rates g (linear interpolation; 0 beyond the spectrum)."""
    return np.interp(g, f, S, right=0.0)


def harmonic_sum(f, Sw, a2=1.0):
    """argmax over f in [LO, HI] of S(f) + a2*S(2f) on the wide-band spectrum (pitch-detection style)."""
    inb = np.where((f >= LO) & (f <= HI))[0]
    H = Sw[inb] + a2 * S_at(f, Sw, 2 * f[inb])
    full = np.zeros_like(Sw); full[inb] = H
    return interp_peak(f, full, inb[np.argmax(H)])


def acf_curve(w):
    w = w - w.mean()
    ac = np.correlate(w, w, "full")[len(w) - 1:]
    return ac / max(ac[0], 1e-20)


def acf_rate(w):
    ac = acf_curve(w)
    lo, hi = int(np.floor(FS * 60 / HI)), int(np.ceil(FS * 60 / LO))
    hi = min(hi, len(ac) - 2)
    pk, _ = signal.find_peaks(ac[:hi + 1])
    pk = pk[pk >= lo]
    if pk.size == 0:
        return float("nan")
    i = pk[np.argmax(ac[pk])]
    a, b, c = ac[i - 1], ac[i], ac[i + 1]
    d = a - 2 * b + c
    lag = i + (0.5 * (a - c) / d if d != 0 else 0.0)
    return float(60 * FS / lag)


def spec_x_acf(w):
    """Score every rate by (normalised spectrum) x (autocorrelation at the matching lag, clipped at 0)."""
    f, S = spectrum(w)
    inb = np.where((f >= LO) & (f <= HI))[0]
    ac = acf_curve(w)
    lags = 60 * FS / f[inb]
    a = np.clip(np.interp(lags, np.arange(len(ac)), ac), 0, None)
    sc = S[inb] / S[inb].max() * a
    full = np.zeros_like(S); full[inb] = sc
    return interp_peak(f, full, inb[np.argmax(sc)])


def pint_rate(w):
    w = signal.savgol_filter(w, int(FS * 1.0) | 1, 3)
    pk, _ = signal.find_peaks(w, distance=max(int(FS * 60 / HI * 0.8), 1), prominence=0.3 * np.std(w))
    if len(pk) < 2:
        return float("nan")
    return float(60 * FS / np.median(np.diff(pk)))


def band_ratio(x):
    return R._band_power_ratio(x, FS, BAND)


def peak_snr(xn):
    """Per-channel share of in-band power within ±1.5 br/min of the channel's own spectral peak."""
    f, S = spectrum(xn)
    inb = (f >= LO) & (f <= HI)
    Sb = S[inb]; fb = f[inb]
    pk = fb[np.argmax(Sb, 0)]
    near = np.abs(fb[:, None] - pk[None, :]) <= 1.5
    return (Sb * near).sum(0) / np.maximum(Sb.sum(0), 1e-20)


# ------------------------------------------------------------------------------------------------- combiners
def zs(x):
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)


def comb_pca(xn, xw):
    sel = np.argsort(band_ratio(xn))[::-1][:TOPK]
    mu, sd = xn[:, sel].mean(0), np.maximum(xn[:, sel].std(0), 1e-12)
    xs = (xn[:, sel] - mu) / sd
    _, _, vt = np.linalg.svd(xs, full_matrices=False)
    v = vt[0]
    return xs @ v, ((xw[:, sel] - xw[:, sel].mean(0)) / sd) @ v


def comb_coh(xn, xw):
    snr = peak_snr(xn)
    sel = np.argsort(snr)[::-1][:TOPK]
    sd = np.maximum(xn[:, sel].std(0), 1e-12)
    xs = (xn[:, sel] - xn[:, sel].mean(0)) / sd
    sign = np.sign(xs.T @ xs[:, 0]); sign[sign == 0] = 1
    wts = sign * snr[sel]
    return xs @ wts, ((xw[:, sel] - xw[:, sel].mean(0)) / sd) @ wts


def comb_rfi(xn, xw, keep=0.5):
    """RespirFi-style sign grouping: cosine similarity, leading eigenvector -> two groups, negate one,
    average the channels with the largest |loading| (top `keep` share)."""
    xs = zs(xn)
    U = xs / np.maximum(np.linalg.norm(xs, axis=0), 1e-12)
    C = U.T @ U
    ev, V = np.linalg.eigh(C)
    v = V[:, -1]
    m = max(2, int(keep * len(v)))
    sel = np.argsort(np.abs(v))[::-1][:m]
    wts = np.sign(v[sel])
    sd = np.maximum(xn[:, sel].std(0), 1e-12)
    wn = ((xn[:, sel] - xn[:, sel].mean(0)) / sd) @ wts
    ww = ((xw[:, sel] - xw[:, sel].mean(0)) / sd) @ wts
    wn = signal.savgol_filter(wn, int(FS * 1.0) | 1, 3)
    return wn, ww


def sum_spectra(xn, xw, k=None):
    sel = np.argsort(band_ratio(xn))[::-1][:(k or TOPK)]
    f, S = spectrum(zs(xn[:, sel]))
    inb = (f >= LO) & (f <= HI)
    Ssum = (S / np.maximum(S[inb].sum(0), 1e-20)).sum(1)
    _, Sw = spectrum(zs(xw[:, sel]))
    Swsum = (Sw / np.maximum(Sw[inb].sum(0), 1e-20)).sum(1)
    return f, Ssum, Swsum


def estimates_for(xn, xw) -> dict:
    out = {}
    f, Ssum, Swsum = sum_spectra(xn, xw)
    out["sum/spec"] = peak_in_band(f, Ssum)
    out["sum/hs"] = harmonic_sum(f, Swsum, 1.0)
    out["sum/hs5"] = harmonic_sum(f, Swsum, 0.5)
    for cname, comb in (("pca", comb_pca), ("coh", comb_coh), ("rfi", comb_rfi)):
        wn, ww = comb(xn, xw)
        fw, Sn = spectrum(wn)
        _, Sww = spectrum(ww)
        out[f"{cname}/spec"] = peak_in_band(fw, Sn)
        out[f"{cname}/acf"] = acf_rate(wn)
        out[f"{cname}/pint"] = pint_rate(wn)
        out[f"{cname}/hs"] = harmonic_sum(fw, Sww, 1.0)
        out[f"{cname}/hs5"] = harmonic_sum(fw, Sww, 0.5)
        out[f"{cname}/sxa"] = spec_x_acf(wn)
    return out


# ------------------------------------------------------------------------------------------- per recording
def process(r: dict, reps_only=None) -> list[dict]:
    g = R.bandpass(signal.detrend(r["gt"].astype(float)), FS, BAND)
    meta = r["meta"]
    reps = representations(r)
    if reps_only:
        reps = {k: v for k, v in reps.items() if k in reps_only}
    pre = {k: (prep(v, BAND), prep(v, WBAND)) for k, v in reps.items()}
    n, h = int(WIN * FS), int(HOP * FS)
    rows = []
    for end in range(n, len(g) + 1, h):
        sl = slice(end - n, end)
        ref = peak_in_band(*spectrum(g[sl]))
        est = {}
        chans = {}
        for k, (xn, xw) in pre.items():
            xn, xw = xn[sl], xw[sl]
            if np.iscomplexobj(xn):
                xn, xw = project(xn, xw)
            good = xn.std(0) > 1e-12
            chans[k] = (xn[:, good], xw[:, good])
        for a_, b_ in (("amp", "cscr1"), ("amp", "san")):
            if a_ in chans and b_ in chans:
                chans[f"{a_}+{b_}"] = tuple(np.concatenate([chans[a_][i], chans[b_][i]], 1) for i in (0, 1))
        for k, (xn, xw) in chans.items():
            for e, v in estimates_for(xn, xw).items():
                est[f"{k}|{e}"] = v
        rows.append(dict(meta=meta, ref=ref, est=est, t_end=end / FS))
    return rows


def _job(r):
    return process(r)


# ---------------------------------------------------------------------------------------- focused follow-up
FOCUS_OUT = ROOT / "data" / "cache" / "repr_focus_rows.pkl"


def window_channels(r, reps_only=("amp", "amp234", "cscr1", "cscr4", "san")):
    """Yield (ref, {name: (narrow, wide) real channels}) per 20 s window of a cached recording."""
    g = R.bandpass(signal.detrend(r["gt"].astype(float)), FS, BAND)
    reps = {k: v for k, v in representations(r).items() if k in reps_only}
    pre = {k: (prep(v, BAND), prep(v, WBAND)) for k, v in reps.items()}
    n, h = int(WIN * FS), int(HOP * FS)
    for end in range(n, len(g) + 1, h):
        sl = slice(end - n, end)
        ref = peak_in_band(*spectrum(g[sl]))
        chans = {}
        for k, (xn, xw) in pre.items():
            xn, xw = xn[sl], xw[sl]
            if np.iscomplexobj(xn):
                xn, xw = project(xn, xw)
            good = xn.std(0) > 1e-12
            chans[k] = (xn[:, good], xw[:, good])
        yield ref, chans


def focus_estimates(chans) -> dict:
    out = {}
    cat = lambda *names: tuple(np.concatenate([chans[nm][i] for nm in names], 1) for i in (0, 1))  # noqa: E731
    pools = {"amp": chans["amp"], "amp234": chans["amp234"], "amp+cscr1": cat("amp", "cscr1"),
             "amp+cscr4": cat("amp", "cscr4"), "amp+san": cat("amp", "san"), "amp+san+cscr1": cat("amp", "san", "cscr1")}
    for name, (xn, xw) in pools.items():
        for k in (10, 20, 30, 45, 60, 1000):
            f, S, _ = sum_spectra(xn, xw, k)
            out[f"{name}|sum{k if k < 1000 else 'all'}"] = peak_in_band(f, S)
    # late fusion: separately summed spectra of each representation, each scaled to unit in-band sum, then added
    for a_, b_ in (("amp", "cscr1"), ("amp", "san"), ("amp", "cscr4")):
        f, Sa, _ = sum_spectra(*chans[a_])
        _, Sb, _ = sum_spectra(*chans[b_])
        inb = (f >= LO) & (f <= HI)
        out[f"late {a_}+{b_}|sum30"] = peak_in_band(f, Sa / Sa[inb].sum() + Sb / Sb[inb].sum())
    return out


def _focus_job(r):
    return [dict(meta=r["meta"], ref=ref, est=focus_estimates(ch)) for ref, ch in window_channels(r)]


# ------------------------------------------------------------------------------------------------- reporting
def table(rows, keys=None, by_subject=True):
    keys = keys or sorted(rows[0]["est"].keys())
    ref = np.array([r["ref"] for r in rows])
    env = np.array([r["meta"]["environment"] for r in rows])
    sub = np.array([r["meta"]["subject"] for r in rows])
    res = []
    for k in keys:
        est = np.array([r["est"][k] for r in rows])
        bad = ~np.isfinite(est)
        est = np.where(bad, 21.0, est)  # a missing estimate counts as the band centre (never dropped)
        e = np.abs(est - ref)
        d = dict(key=k, mae=e.mean(), w2=np.mean(e <= 2), quiet=e[env == "Quiet"].mean(),
                 interf=e[env == "Interference"].mean(), nan=bad.mean())
        for s in sorted(set(sub)):
            d[s] = e[sub == s].mean()
        res.append(d)
    return sorted(res, key=lambda d: d["mae"])


def print_table(res, n=None):
    subs = [k for k in res[0] if k.startswith("S")]
    print(f"{'variant':26s}  MAE    ±2   Quiet  Interf  " + "  ".join(f"{s:>5s}" for s in subs) + "  nan")
    for d in res[:n]:
        print(f"{d['key']:26s} {d['mae']:5.2f} {d['w2']:5.1%} {d['quiet']:6.2f} {d['interf']:6.2f}  "
              + "  ".join(f"{d[s]:5.2f}" for s in subs) + f"  {d['nan']:.0%}")


# ------------------------------------------------------------------------------------------------- synthetic
def synth_recording(bpm, rng, complex_mode, dur=30.0, snr_db=0.0, excursion=0.8):
    """Raw per-packet CSI like WiRe-Breath: ~49 Hz jittered packets, 234 columns of which positions 0-52 hold
    the channel (60-ish in the task spec; here the real layout: 52 + a null at 25) and the rest are junk.
    amplitude mode: breathing modulates |H| with random sign/depth per subcarrier.
    complex mode: static path + chest path whose phase follows the chest; every packet gets a random common
    phase (CFO), a random phase slope over subcarriers (timing offset) and common gain noise (AGC)."""
    t = np.cumsum(rng.uniform(0.018, 0.023, int(dur * 49)))
    t -= t[0]
    br = np.sin(2 * np.pi * bpm / 60 * t + rng.uniform(0, 2 * np.pi))
    k = np.arange(53)
    if not complex_mode:
        A = rng.uniform(500, 2000, 53)
        d = rng.choice([-1, 1], 53) * rng.uniform(0.02, 0.1, 53) * A
        noise_sd = np.sqrt(np.mean(d ** 2) / 2) * 10 ** (-snr_db / 20)
        amp = A + d * br[:, None] + rng.normal(0, 1, (len(t), 53)) * noise_sd
        H = amp * np.exp(1j * np.cumsum(rng.uniform(-0.3, 0.3, 53)))[None, :]
    else:
        stat = rng.uniform(500, 2000, 53) * np.exp(1j * rng.uniform(0, 2 * np.pi, 53))
        dyn = rng.uniform(50, 200, 53) * np.exp(1j * rng.uniform(0, 2 * np.pi, 53))
        kk = rng.uniform(0.8, 1.2, 53)
        H = stat + dyn * np.exp(1j * kk[None, :] * excursion * br[:, None])
        noise_sd = np.mean(np.abs(dyn)) * 0.4 * 10 ** (-snr_db / 20)
        H = H + noise_sd * (rng.normal(size=H.shape) + 1j * rng.normal(size=H.shape)) / np.sqrt(2)
    a = rng.uniform(0, 2 * np.pi, (len(t), 1))
    b = rng.normal(0, 0.3, (len(t), 1))
    gain = 1 + 0.05 * rng.normal(size=(len(t), 1))
    H = H * np.exp(1j * (a + b * k[None, :])) * gain
    H[:, 25] *= 0.03
    junk = rng.integers(-5, 6, (len(t), 181)) + 1j * rng.integers(-5, 6, (len(t), 181))
    c = np.concatenate([H, junk], 1)
    return build_from_arrays(t, c, t, br, dict(environment="synth", subject="S0"))


def synth_check(seed=0):
    rng = np.random.default_rng(seed)
    print("\nSYNTHETIC CHECK (|error| br/min, max over the 3 windows of a 30 s recording; pass = all <= 0.5)")
    allres = {}
    for mode in (False, True):
        for bpm in (7, 12, 22, 30):
            rows = process(synth_recording(bpm, rng, mode))
            for k in rows[0]["est"]:
                e = max(abs(r["est"][k] - bpm) if np.isfinite(r["est"][k]) else 99 for r in rows)
                allres[("complex" if mode else "amplitude-only", bpm, k)] = e
            print(f"  mode={'complex+phase offset' if mode else 'amplitude      '} bpm={bpm:2d}  ref err "
                  f"{max(abs(r['ref'] - bpm) for r in rows):.2f}")
    keys = sorted({k[2] for k in allres})
    print(f"  {'variant':26s} " + " ".join(f"{m[:4]}{b:>3d}" for m in ("amp", "cplx") for b in (7, 12, 22, 30)))
    fails = {}
    for k in keys:
        vals = [allres[(m, b, k)] for m in ("amplitude-only", "complex") for b in (7, 12, 22, 30)]
        fails[k] = vals
        print(f"  {k:26s} " + " ".join(f"{v:7.2f}" for v in vals))
    return fails


# ------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--focus", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.synth:
        synth_check()
        return
    if a.focus:
        from joblib import Parallel, delayed
        if FOCUS_OUT.exists() and not a.recompute:
            rows = pickle.loads(FOCUS_OUT.read_bytes())
        else:
            recs = list(np.load(CACHE, allow_pickle=True))
            rows = [w for o in Parallel(n_jobs=a.jobs, verbose=2)(delayed(_focus_job)(r) for r in recs) for w in o]
            FOCUS_OUT.write_bytes(pickle.dumps(rows))
        print_table(table(rows))
        return
    if OUT.exists() and not a.recompute and not a.limit:
        rows = pickle.loads(OUT.read_bytes())
    else:
        from joblib import Parallel, delayed
        if not CACHE.exists():
            build_cache(a.jobs)
        recs = list(np.load(CACHE, allow_pickle=True))
        if a.limit:
            recs = recs[:: max(1, len(recs) // a.limit)][:a.limit]
        out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(_job)(r) for r in recs)
        rows = [w for o in out for w in o]
        if not a.limit:
            OUT.write_bytes(pickle.dumps(rows))
    print(f"{len(rows)} windows")
    print_table(table(rows))


if __name__ == "__main__":
    main()
