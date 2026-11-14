"""Respiration-rate estimation from single-antenna Wi-Fi CSI.

Known building blocks, no learned rate model:
  per-packet phase sanitisation (in the loader) -> Hampel -> detrend -> band-pass 0.1-0.6 Hz (6-36 br/min),
  filtered once over the whole recording
  -> channels: subcarrier amplitude + cross-subcarrier ratio H(m)/H(ref) projected onto its max-variance axis
  -> per 20 s window: top-30 channels by breathing-band power ratio, summed normalised spectra
  -> rate = a spectral peak, linked across windows by Viterbi tracking (can only choose between real peaks)
  -> confidence = calibrated P(within ±2 br/min) from peak prominence, x a coherence guard (noise -> 0).
The classic first-principal-component peak (``estimate_classic``) is kept as the reference baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

BAND_HZ = (0.1, 0.6)          # estimator band (Hz)
REF_BAND_HZ = (0.05, 0.8)     # chest-belt reference: wider and fixed, so it stays right at slow/fast rates
CALIB_PATH = Path(__file__).resolve().parents[2] / "models" / "breath_confidence.joblib"


@dataclass
class BreathEstimate:
    bpm: float              # estimated breaths per minute (nan if no estimate)
    confidence: float       # 0..1, spectral peak prominence + method agreement
    waveform: np.ndarray    # band-passed breathing waveform (z-scored)
    bpm_spectral: float
    bpm_peaks: float
    n_subcarriers: int
    ambiguous: bool = False   # a wider band sees half this rate: possibly slow breathing read via its harmonic


def hampel(x: np.ndarray, k: int = 15, n_sigmas: float = 3.0) -> np.ndarray:
    """Vectorised Hampel filter along axis 0 (works for (T,) or (T, S))."""
    from scipy.ndimage import median_filter
    x = np.asarray(x, dtype=float)
    size = (2 * k + 1,) + (1,) * (x.ndim - 1)
    med = median_filter(x, size=size, mode="nearest")
    mad = median_filter(np.abs(x - med), size=size, mode="nearest") * 1.4826
    out = x.copy()
    bad = np.abs(x - med) > n_sigmas * np.maximum(mad, 1e-12)
    out[bad] = med[bad]
    return out


def bandpass(x: np.ndarray, fs: float, band=None, order: int = 4) -> np.ndarray:
    band = BAND_HZ if band is None else band
    sos = signal.butter(order, [band[0], band[1]], btype="band", fs=fs, output="sos")
    # scipy default "odd" padding mirrors around the first sample, which on noisy CSI is random and injects a
    # strong start/end transient; mirror ("even") padding over up to 3 periods of the low cut-off avoids it
    padlen = min(x.shape[0] - 1, int(3 * fs / band[0]))
    return signal.sosfiltfilt(sos, x, axis=0, padtype="even", padlen=padlen)


def _band_power_ratio(x: np.ndarray, fs: float, band=None) -> np.ndarray:
    band = BAND_HZ if band is None else band
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 20)), axis=0)
    inb = (f >= band[0]) & (f <= band[1])
    tot = (f > 0.02) & (f <= min(5.0, fs / 2))
    return p[inb].sum(0) / np.maximum(p[tot].sum(0), 1e-20)


def spectral_rate(x: np.ndarray, fs: float, band=None, pad_factor: int = 16):
    """Return (bpm, prominence) using a zero-padded FFT peak with parabolic interpolation."""
    band = BAND_HZ if band is None else band
    x = (x - x.mean()) * np.hanning(len(x))
    n = int(2 ** np.ceil(np.log2(len(x) * pad_factor)))
    spec = np.abs(np.fft.rfft(x, n))
    f = np.fft.rfftfreq(n, 1 / fs)
    inb = np.where((f >= band[0]) & (f <= band[1]))[0]
    if inb.size < 3:
        return float("nan"), 0.0
    i = inb[np.argmax(spec[inb])]
    if 0 < i < len(spec) - 1:
        a, b, c = spec[i - 1], spec[i], spec[i + 1]
        delta = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    else:
        delta = 0.0
    f_peak = f[i] + delta * (f[1] - f[0])
    p = spec ** 2
    near = inb[np.abs(f[inb] - f[i]) <= 0.04]
    prominence = float(p[near].sum() / np.maximum(p[inb].sum(), 1e-20))
    return float(f_peak * 60.0), prominence


def peak_count_rate(x: np.ndarray, fs: float, band=None) -> float:
    band = BAND_HZ if band is None else band
    min_dist = int(fs / band[1] * 0.8)
    prom = 0.3 * np.std(x)
    peaks, _ = signal.find_peaks(x, distance=max(min_dist, 1), prominence=prom)
    if len(peaks) < 2:
        return float("nan")
    return float(60.0 * fs / np.median(np.diff(peaks)))


def preprocess(amp: np.ndarray, fs: float, band=None) -> np.ndarray:
    """Hampel -> detrend -> band-pass, applied to a whole recording / long stream buffer.

    Filtering each short window on its own creates edge transients at 0.1 Hz that swamp the breathing
    tone, so callers filter once and cut windows from the result (see ``sliding``)."""
    amp = np.asarray(amp, dtype=float)
    if amp.ndim == 1:
        amp = amp[:, None]
    good = np.std(amp, axis=0) > 1e-9
    amp = amp[:, good] if good.any() else amp
    x = hampel(amp)
    x = signal.detrend(x, axis=0)
    return bandpass(x, fs, band)


def waveform_from_filtered(x: np.ndarray, fs: float, top_k: int = 30):
    """Band-passed CSI window (T, S) -> (breathing waveform (T,), n_selected)."""
    ratio = _band_power_ratio(x, fs)
    k = min(top_k, x.shape[1])
    sel = np.argsort(ratio)[::-1][:k]
    xs = x[:, sel]
    xs = (xs - xs.mean(0)) / np.maximum(xs.std(0), 1e-12)
    u, s, _ = np.linalg.svd(xs, full_matrices=False)  # first principal component
    w = u[:, 0] * s[0]
    w = (w - w.mean()) / max(w.std(), 1e-12)
    return w, k


def breathing_waveform(amp: np.ndarray, fs: float, top_k: int = 30):
    """CSI amplitude (T, S) -> (waveform (T,), n_selected)."""
    return waveform_from_filtered(preprocess(amp, fs), fs, top_k)


def estimate_classic(x: np.ndarray, fs: float) -> BreathEstimate:
    """Spectral peak of the first principal component, confidence from peak prominence (no learning)."""
    w, k = waveform_from_filtered(x, fs)
    bpm_s, prom = spectral_rate(w, fs)
    bpm_p = peak_count_rate(w, fs)
    agree = 1.0
    if np.isfinite(bpm_p) and np.isfinite(bpm_s):
        agree = float(np.exp(-abs(bpm_s - bpm_p) / 3.0))
    # prominence of pure noise sits around 0.2-0.3; a clean breathing tone is ~0.9+
    conf = float(np.clip((prom - 0.25) / 0.6, 0, 1) * (0.5 + 0.5 * agree))
    return BreathEstimate(bpm_s, conf, w, bpm_s, bpm_p, k)


@lru_cache(maxsize=1)
def load_calibration():
    """Confidence calibration (scripts/train_breath_confidence.py), or None -> uncalibrated prominence score."""
    if not CALIB_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(CALIB_PATH)
    except Exception:  # noqa: BLE001 - a broken/foreign model file must not take the pipeline down
        return None


# ------------------------------------------------------------------ fused amplitude + phase estimator
FINE = np.arange(6.0, 36.01, 0.1)       # br/min grid of the summed spectrum
TRACK_LAM = 0.15                        # log-score cost per br/min of jump between windows 5 s apart
COHERENCE = (0.20, 0.26)
WIDE_BAND_HZ = (0.05, 0.6)              # ambiguity check: sees down to ~3 br/min
WIDE_GRID = np.arange(4.0, 36.01, 0.1)
AMBIGUITY_TOL = 0.12                    # wide-band rate within 0.5 +- this of the reported rate                # PC1 variance share: independent noise < 0.19, real breathing > 0.24


@dataclass
class Channels:
    """Band-passed breathing channels of a whole recording / stream buffer."""
    amp: np.ndarray                     # (T, Sa) filtered amplitude
    ratio: np.ndarray | None            # (T, Sc) filtered complex cross-subcarrier ratio, or None


def channels(csi: np.ndarray, fs: float, band=None) -> Channels:
    """Complex CSI (T, S) with sanitised phase (see ingest.wirebreath.sanitize_phase), or real amplitude.

    The cross-subcarrier ratio H(m)/H(ref) of one packet cancels what all subcarriers share (residual phase
    offset, AGC gain), leaving breathing-modulated phase on a single antenna (SA-WiSense, arXiv 2507.17623)."""
    csi = np.asarray(csi)
    if not np.iscomplexobj(csi):
        return Channels(preprocess(csi, fs, band), None)
    amp = np.abs(csi)
    ref = int(np.argmax(amp.mean(0) / (amp.std(0) + 1e-9)))
    r = np.delete(csi, ref, axis=1) / csi[:, [ref]]
    filt = preprocess(np.concatenate([r.real, r.imag], 1), fs, band)
    k = r.shape[1]
    ratio = filt[:, :k] + 1j * filt[:, k:2 * k] if filt.shape[1] == 2 * k else None
    return Channels(preprocess(amp, fs, band), ratio)


def _project(z: np.ndarray) -> np.ndarray:
    """Each complex column onto its maximum-variance axis (the FarSense-style best projection), per window."""
    re, im = z.real - z.real.mean(0), z.imag - z.imag.mean(0)
    th = 0.5 * np.arctan2(2 * (re * im).mean(0), (re ** 2).mean(0) - (im ** 2).mean(0))
    return re * np.cos(th) + im * np.sin(th)


def _spectra(x: np.ndarray, fs: float, pad: int = 16):
    x = x - x.mean(0)
    w = np.hanning(len(x))
    x = x * (w[:, None] if x.ndim == 2 else w)
    n = int(2 ** np.ceil(np.log2(len(x) * pad)))
    return np.fft.rfftfreq(n, 1 / fs) * 60.0, np.abs(np.fft.rfft(x, n, axis=0)) ** 2


@dataclass
class WindowSpectrum:
    fine: np.ndarray                    # summed normalised spectrum on FINE (max = 1)
    waveform: np.ndarray                # first principal component of the selected channels (z-scored)
    coherence: float                    # PC1 variance share of the selected channels
    n: int


def window_spectrum(ch: Channels, s: slice, fs: float, top_k: int = 30, grid: np.ndarray | None = None) -> WindowSpectrum:
    grid = FINE if grid is None else grid
    x = ch.amp[s]
    if ch.ratio is not None:
        x = np.concatenate([x, _project(ch.ratio[s])], 1)
    ratio = _band_power_ratio(x, fs)
    sel = np.argsort(ratio)[::-1][:min(top_k, x.shape[1])]
    xs = x[:, sel]
    xs = (xs - xs.mean(0)) / np.maximum(xs.std(0), 1e-12)
    fb, S = _spectra(xs, fs)
    inb = (fb >= grid[0]) & (fb <= grid[-1])
    Ssum = (S / np.maximum(S[inb].sum(0), 1e-20)).sum(1)
    fine = np.interp(grid, fb, Ssum)
    fine = fine / max(fine.max(), 1e-20)
    u, sv, _ = np.linalg.svd(xs, full_matrices=False)
    w = u[:, 0] * sv[0]
    return WindowSpectrum(fine, (w - w.mean()) / max(w.std(), 1e-12),
                          float(sv[0] ** 2 / max(np.sum(sv ** 2), 1e-20)), len(sel))


def _peaks_only(fine: np.ndarray, floor: float = 0.25) -> np.ndarray:
    """Candidate scores: spectral peaks with real support keep their height, everything else ~0. Tracking can
    therefore only choose between genuine peaks; it never slides a reading off a peak toward 'typical' rates."""
    pad = np.concatenate([[-1.0], fine, [-1.0]])
    peak = (pad[1:-1] >= pad[:-2]) & (pad[1:-1] >= pad[2:]) & (fine >= floor)
    return np.where(peak, fine, 1e-6)


def _parabolic(fine: np.ndarray, j: int, grid: np.ndarray | None = None) -> float:
    FINE_ = FINE if grid is None else grid
    if 0 < j < len(fine) - 1:
        a, b, c = fine[j - 1], fine[j], fine[j + 1]
        d = a - 2 * b + c
        return float(FINE_[j] + (0.5 * (a - c) / d if d != 0 else 0.0) * (FINE_[1] - FINE_[0]))
    return float(FINE_[j])


def prominence(fine: np.ndarray, bpm: float, half: float = 2.4) -> float:
    """Share of in-band spectral power within ±half br/min of the reading."""
    return float(fine[np.abs(FINE - bpm) <= half].sum() / max(fine.sum(), 1e-20))


def confidence_features(ws: WindowSpectrum, bpm: float) -> np.ndarray:
    guard = float(np.clip((ws.coherence - COHERENCE[0]) / (COHERENCE[1] - COHERENCE[0]), 0, 1))
    return np.array([prominence(ws.fine, bpm), guard])


def confidence(ws: WindowSpectrum, bpm: float) -> float:
    """Calibrated P(reading within ±2 br/min) when the calibration exists; noise (incoherent) -> 0."""
    prom, guard = confidence_features(ws, bpm)
    cal = load_calibration()
    p = float(cal["iso"].predict([prom])[0]) if cal is not None else float(np.clip((prom - 0.2) / 0.5, 0, 1))
    return p * guard


def track(scores: list[np.ndarray], lam: float, grid: np.ndarray | None = None) -> list[int]:
    """Viterbi path through per-window candidate scores; each br/min of jump costs ``lam`` (log units)."""
    if not scores:
        return []
    L = [np.log(np.clip(s, 1e-6, None)) for s in scores]
    g = FINE if grid is None else grid
    jump = lam * np.abs(g[:, None] - g[None, :])
    score, back = L[0].copy(), []
    for l in L[1:]:
        cand = score[:, None] - jump
        b = np.argmax(cand, 0)
        back.append(b)
        score = cand[b, np.arange(len(g))] + l
    path = [int(np.argmax(score))]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]


def _make(ws: WindowSpectrum, j: int, fs: float, wide_bpm: float | None = None) -> BreathEstimate:
    bpm = _parabolic(ws.fine, j)
    amb = wide_bpm is not None and abs(wide_bpm / bpm - 0.5) < AMBIGUITY_TOL
    conf = 0.0 if amb else confidence(ws, bpm)
    return BreathEstimate(bpm, conf, ws.waveform, bpm, peak_count_rate(ws.waveform, fs), ws.n, bool(amb))


def estimate_channels(ch: Channels, fs: float, s: slice | None = None) -> BreathEstimate:
    ws = window_spectrum(ch, s or slice(None), fs)
    return _make(ws, int(np.argmax(_peaks_only(ws.fine))), fs)


def estimate_filtered(x: np.ndarray, fs: float) -> BreathEstimate:
    """One band-passed amplitude window -> estimate (fused spectrum without the phase channels)."""
    return estimate_channels(Channels(x, None), fs)


def estimate(csi: np.ndarray, fs: float) -> BreathEstimate:
    """Single self-contained window of complex CSI or amplitude (prefer ``sliding`` on long recordings)."""
    n = np.asarray(csi).shape[0]
    return list(sliding(csi, fs, win_s=n / fs, hop_s=n / fs))[-1][1]


def filtered_reference(gt_time: np.ndarray, gt_signal: np.ndarray):
    """Reference respiration resampled to a uniform grid and band-passed once: (t, x, fs)."""
    t = np.asarray(gt_time, float)
    s = np.asarray(gt_signal, float)
    fs = 1.0 / np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], 1 / fs)
    x = bandpass(signal.detrend(np.interp(tu, t, s)), fs, REF_BAND_HZ)
    return tu, x, fs


def reference_rate(gt_time: np.ndarray, gt_signal: np.ndarray, t0: float | None = None,
                   t1: float | None = None) -> float:
    """Breathing rate from the reference signal over [t0, t1] (filtered over the whole recording first)."""
    tu, x, fs = filtered_reference(gt_time, gt_signal)
    m = np.ones_like(tu, bool)
    if t0 is not None:
        m &= tu >= t0
    if t1 is not None:
        m &= tu <= t1
    if m.sum() < 10:
        return float("nan")
    return spectral_rate(x[m], fs, REF_BAND_HZ)[0]


def sliding(csi: np.ndarray, fs: float, win_s: float = 20.0, hop_s: float = 1.0, tracking: bool = True,
            ambiguity_check: bool = True):
    """Yield (t_end_seconds, BreathEstimate) over sliding windows of a recording / stream buffer.

    Accepts complex CSI (amplitude + phase channels) or real amplitude. With ``tracking`` the windows are
    linked by a Viterbi path over spectral peaks, so a single window's stray harmonic cannot flip the reading.
    With ``ambiguity_check`` a second, wider-band estimate runs alongside: when it sees half the reported rate,
    the reading may be slow breathing seen through its 2nd harmonic, so its confidence is set to 0 (shown as
    unclear, never as a confident normal rate). On WiRe-Breath slowed to 0.6x this cut confident readings off by
    >3 br/min from 13.3% to 3.8%, at no cost in accuracy (scripts/exp_slow_breathing.py ambiguity)."""
    ch = channels(csi, fs)
    n, h = int(win_s * fs), max(int(hop_s * fs), 1)
    ends = list(range(n, ch.amp.shape[0] + 1, h))
    lam = TRACK_LAM * 5.0 / max(hop_s, 1e-6)
    wss = [window_spectrum(ch, slice(e - n, e), fs) for e in ends]
    scores = [_peaks_only(w.fine) for w in wss]
    path = track(scores, lam) if tracking else [int(np.argmax(sc)) for sc in scores]
    wide = [None] * len(ends)
    if ambiguity_check and ends:
        chw = channels(csi, fs, WIDE_BAND_HZ)
        wsw = [window_spectrum(chw, slice(e - n, e), fs, grid=WIDE_GRID) for e in ends]
        sw = [_peaks_only(w.fine) for w in wsw]
        pw = track(sw, lam, WIDE_GRID) if tracking else [int(np.argmax(sc)) for sc in sw]
        wide = [_parabolic(w.fine, j, WIDE_GRID) for w, j in zip(wsw, pw)]
    for e, ws, j, wb in zip(ends, wss, path, wide):
        yield e / fs, _make(ws, j, fs, wb)
