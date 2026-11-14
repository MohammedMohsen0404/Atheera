"""Abnormal-motion detection.

Three explainable signals, combined into one score:
  1. motion index   — high-frequency CSI amplitude energy (body movement), z-scored against a
                      per-astronaut running baseline (median / MAD), so "abnormal" is relative to that
                      person's own normal in that module;
  2. fall prob      — from the activity model (SenseFi UT-HAR 'fall' class);
  3. stillness rule — occupied + no motion + no credible breathing signal (handled in fusion).
Optionally an IsolationForest over [motion, bpm, confidence] learned on normal windows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal


def motion_index(amp: np.ndarray, fs: float, cutoff_hz: float = 0.8, gain_norm: bool = False) -> float:
    """Mean high-pass energy across subcarriers, normalised by mean amplitude (scale-free).

    gain_norm (linear amplitude only): divide each packet by its median across subcarriers first. Commodity
    Wi-Fi receivers change their gain (AGC) in steps that scale every subcarrier at once; without this, one
    step reads as ~3 s of strong body motion. Movement changes the pattern across subcarriers, which stays.
    On WiRe-Breath still recordings this cut seconds above 4x the typical motion from 5.4% to 0.5%, while a
    person moving nearby still raised 29% of seconds above that level.
    """
    amp = np.asarray(amp, float)
    if amp.ndim == 1:
        amp = amp[:, None]
    if gain_norm and amp.shape[1] > 1:
        g = np.median(amp, axis=1, keepdims=True)
        amp = amp / np.maximum(g, 1e-9) * np.median(g)
    if amp.shape[0] < 16:
        return 0.0
    cutoff = min(cutoff_hz, 0.45 * fs)
    sos = signal.butter(3, cutoff, btype="high", fs=fs, output="sos")
    hp = signal.sosfiltfilt(sos, amp - amp.mean(0), axis=0)
    return float(np.mean(np.std(hp, axis=0)) / max(np.mean(np.abs(amp)), 1e-9))


@dataclass
class AnomalyResult:
    score: float        # 0..1
    motion_z: float
    reasons: list[str]


class AnomalyDetector:
    def __init__(self, history: int = 120, warmup: int = 15):
        self.hist: deque[float] = deque(maxlen=history)
        self.warmup = warmup
        self.iforest = None

    def fit_iforest(self, X: np.ndarray):
        from sklearn.ensemble import IsolationForest
        self.iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(X)
        return self

    def update(self, motion: float, fall_prob: float = 0.0, bpm: float = np.nan, conf: float = 0.0) -> AnomalyResult:
        reasons = []
        if len(self.hist) >= self.warmup:
            h = np.asarray(self.hist)
            med = np.median(h)
            # floor the spread at half the typical level: for a very still sleeper the MAD collapses toward 0
            # and tiny wobbles would read as many MADs; with the floor, z = 6 needs ~4x the usual motion
            mad = max(np.median(np.abs(h - med)) * 1.4826, 0.5 * abs(med)) + 1e-6
            z = (motion - med) / mad
        else:
            z = 0.0
        self.hist.append(motion)
        s_motion = float(1 / (1 + np.exp(-(z - 6) / 1.5)))  # ~0 below 3 MAD, ~1 above 9 MAD
        if s_motion > 0.5:
            reasons.append(f"motion {z:.1f} MAD above personal baseline")
        s_fall = float(fall_prob)
        if s_fall > 0.5:
            reasons.append(f"activity model: fall p={fall_prob:.2f}")
        s_if = 0.0
        if self.iforest is not None and np.isfinite(bpm):
            raw = -self.iforest.score_samples([[motion, bpm, conf]])[0]
            s_if = float(np.clip((raw - 0.5) / 0.2, 0, 1))
            if s_if > 0.5:
                reasons.append("unusual motion/breathing combination (IsolationForest)")
        score = float(max(s_fall, 1 - (1 - s_motion) * (1 - 0.5 * s_if)))
        return AnomalyResult(score, float(z), reasons)
