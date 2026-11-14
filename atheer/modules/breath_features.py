"""EXPERIMENTS ONLY - not used by the shipped pipeline (atheer/modules/respiration.py).

Kept because scripts/exp_breath_nn.py (the neural-model study, which did not beat the shipped non-learned
estimator) imports it. Candidate-rate features for the (rejected) learned breathing peak picker.

Every rate on a 0.5 br/min grid (6-36 br/min) is described by how strongly the band-passed CSI supports
it: summed per-subcarrier spectrum, principal-component spectrum, how many subcarriers peak there,
power at its harmonic and sub-harmonic, autocorrelation at the matching lag, and peak-count agreement.
Pure functions of one window of already band-passed CSI (see respiration.preprocess).
"""
from __future__ import annotations

import numpy as np

from .respiration import BAND_HZ, _band_power_ratio, peak_count_rate

GRID = np.arange(6.0, 36.01, 0.5)  # br/min
FEATURES = ["sum", "sum_rank", "pca", "vote", "wvote", "h2", "h_half", "acf", "local_max", "peakcount_gap",
            "win_prom", "win_ratio", "pc1_share"]


def _spec(x: np.ndarray, fs: float, pad: int = 8):
    x = x - x.mean(0)
    w = np.hanning(len(x))
    x = x * (w[:, None] if x.ndim == 2 else w)
    n = int(2 ** np.ceil(np.log2(len(x) * pad)))
    return np.fft.rfftfreq(n, 1 / fs) * 60.0, np.abs(np.fft.rfft(x, n, axis=0)) ** 2


def _at(fb: np.ndarray, S: np.ndarray, bpm: np.ndarray, half: float = 0.6) -> np.ndarray:
    """Max of spectrum S (1-D) within ±half br/min of each requested rate (0 outside the spectrum)."""
    out = np.zeros(len(bpm))
    for i, b in enumerate(bpm):
        m = np.abs(fb - b) <= half
        out[i] = S[m].max() if m.any() else 0.0
    return out


FINE = np.arange(6.0, 36.01, 0.1)  # br/min, for refining the chosen candidate


def candidate_features(x: np.ndarray, fs: float, top_k: int = 30, return_fine: bool = False):
    """Band-passed window (T, S) -> (features [len(GRID), len(FEATURES)], GRID[, summed spectrum on FINE])."""
    ratio = _band_power_ratio(x, fs)
    k = min(top_k, x.shape[1])
    sel = np.argsort(ratio)[::-1][:k]
    xs = x[:, sel]
    xs = (xs - xs.mean(0)) / np.maximum(xs.std(0), 1e-12)
    fb, S = _spec(xs, fs)
    inb = (fb >= 60 * BAND_HZ[0]) & (fb <= 60 * BAND_HZ[1])
    Sn = S / np.maximum(S[inb].sum(0), 1e-20)
    Ssum = Sn.sum(1); Ssum = Ssum / max(Ssum[inb].max(), 1e-20)
    u, s, _ = np.linalg.svd(xs, full_matrices=False)
    pc = u[:, 0]
    pc1_share = float(s[0] ** 2 / max(np.sum(s ** 2), 1e-20))  # coherence across subcarriers
    _, Sp = _spec(pc, fs); Sp = Sp / max(Sp[inb].max(), 1e-20)
    pk = fb[inb][np.argmax(S[inb], 0)]                       # each subcarrier's own peak rate
    wr = ratio[sel] / max(ratio[sel].sum(), 1e-20)
    ac = np.correlate(pc - pc.mean(), pc - pc.mean(), "full")[len(pc) - 1:]
    ac = ac / max(ac[0], 1e-20)
    pcr = peak_count_rate(pc, fs)

    f_sum = _at(fb, Ssum, GRID)
    order = np.argsort(-f_sum); rank = np.empty_like(order); rank[order] = np.arange(len(GRID))
    vote = np.array([np.mean(np.abs(pk - b) <= 1.0) for b in GRID])
    wvote = np.array([wr[np.abs(pk - b) <= 1.0].sum() for b in GRID])
    lag = np.clip(np.round(60.0 * fs / GRID).astype(int), 1, len(ac) - 1)
    near = np.abs(fb[inb] - fb[inb][np.argmax(Ssum[inb])]) <= 2.4
    win_prom = float(Ssum[inb][near].sum() / max(Ssum[inb].sum(), 1e-20))
    pad = np.concatenate([[-1.0], f_sum, [-1.0]])          # band edges count as peaks if the spectrum rises to them
    loc = ((pad[1:-1] >= pad[:-2]) & (pad[1:-1] >= pad[2:])).astype(float)
    F = np.column_stack([
        f_sum, rank / len(GRID), _at(fb, Sp, GRID), vote, wvote,
        # harmonic / sub-harmonic power only where it can exist inside the band; elsewhere "missing" (NaN),
        # otherwise a structural zero above 18 br/min would tell the model the rate and act as a prior
        np.where(2 * GRID <= 60 * BAND_HZ[1], _at(fb, Ssum, 2 * GRID), np.nan),
        np.where(GRID / 2 >= 60 * BAND_HZ[0], _at(fb, Ssum, GRID / 2), np.nan), ac[lag], loc,
        np.abs(GRID - pcr) if np.isfinite(pcr) else np.full(len(GRID), 30.0),
        np.full(len(GRID), win_prom), np.full(len(GRID), float(np.median(ratio[sel]))),
        np.full(len(GRID), pc1_share),
    ])
    if return_fine:
        return F, GRID, np.interp(FINE, fb, Ssum)
    return F, GRID


def refine(grid_bpm: float, fine: np.ndarray, half: float = 1.0) -> float:
    """Exact rate: the summed-spectrum maximum within +-half br/min of the chosen grid candidate."""
    m = np.abs(FINE - grid_bpm) <= half + 1e-9
    return float(FINE[m][np.argmax(fine[m])])



# acf: autocorrelation at the candidate's lag varies with the rate itself -> acts as a rate prior
# pc1_share: used only by the noise guard, not by the model
EXCLUDED = {"acf", "pc1_share"}


def model_input(F: np.ndarray) -> np.ndarray:
    return F[:, [i for i, f in enumerate(FEATURES) if f not in EXCLUDED]]


# Noise guard: breathing moves many subcarriers together, independent noise does not. On WiRe-Breath 98% of
# windows have a first-component variance share above 0.24; independent noise on 30 subcarriers stays below 0.19.
GUARD = (0.20, 0.26)


def noise_guard(F: np.ndarray) -> float:
    """0 (incoherent, i.e. noise) .. 1 (coherent) multiplier for the learned confidence."""
    share = F[0, FEATURES.index("pc1_share")]
    return float(np.clip((share - GUARD[0]) / (GUARD[1] - GUARD[0]), 0.0, 1.0))


SUPPORT_MIN = 0.25  # a candidate needs at least this much of the strongest in-band spectral peak


def support(F: np.ndarray, thr: float | None = None) -> np.ndarray:
    """1 for candidates that are spectral peaks with real support, else 0. The learned model only chooses
    BETWEEN peaks (true rate vs harmonic vs interference); it can never move a reading off a peak, so a clean
    breathing tone at any rate, including abnormal ones, is always reported where it is."""
    return ((F[:, 0] >= (SUPPORT_MIN if thr is None else thr)) & (F[:, 8] > 0)).astype(float)


def track(probs: list[np.ndarray], grid: np.ndarray = GRID, lam: float = 0.15) -> list[int]:
    """Viterbi path through per-window candidate probabilities; each br/min of jump costs ``lam`` (log units)."""
    L = [np.log(np.clip(p, 1e-6, 1.0)) for p in probs]
    if not L:
        return []
    jump = lam * np.abs(grid[:, None] - grid[None, :])
    score, back = L[0].copy(), []
    for l in L[1:]:
        cand = score[:, None] - jump          # from i -> to j
        b = np.argmax(cand, 0); back.append(b)
        score = cand[b, np.arange(len(grid))] + l
    path = [int(np.argmax(score))]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]
