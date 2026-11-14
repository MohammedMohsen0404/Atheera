import numpy as np
import pytest
from atheer.modules import respiration as R


def synth_csi(bpm, fs=50.0, dur=30.0, S=60, snr=0.6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    br = np.sin(2 * np.pi * bpm / 60 * t)
    gains = rng.uniform(-1, 1, S)
    amp = 20 + np.outer(br, gains) * snr + rng.normal(0, 0.5, (t.size, S))
    amp[rng.integers(0, t.size, 20), rng.integers(0, S, 20)] += 30  # impulsive outliers
    return t, amp


def classic(amp, fs=50.0):
    return R.estimate_classic(R.preprocess(amp, fs), fs)


@pytest.mark.parametrize("bpm", [8, 12, 16, 22, 30])
def test_classic_synthetic_rate(bpm):
    est = classic(synth_csi(bpm)[1])
    assert abs(est.bpm - bpm) < 0.5
    assert est.confidence > 0.5


def test_classic_noise_low_confidence():
    rng = np.random.default_rng(1)
    assert classic(20 + rng.normal(0, 1, (1500, 60))).confidence < classic(synth_csi(15)[1]).confidence


def synth_complex(bpm, fs=50.0, dur=30.0, S=52, seed=0, excursion=0.3):
    """Complex CSI whose breathing shows mostly in PHASE, plus a random per-packet common phase (sanitised)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    br = np.sin(2 * np.pi * bpm / 60 * t)
    static = rng.uniform(5, 10, S) * np.exp(1j * rng.uniform(0, 2 * np.pi, S))
    dyn = rng.uniform(0.5, 1.0, S) * np.exp(1j * rng.uniform(0, 2 * np.pi, S))
    h = static + dyn * np.exp(1j * excursion * br[:, None]) + 0.05 * (rng.normal(size=(t.size, S)) + 1j * rng.normal(size=(t.size, S)))
    return h


@pytest.mark.parametrize("bpm", [7, 8, 12, 16, 22, 30, 34])
def test_no_rate_prior_amplitude(bpm):
    """A clean breathing tone must be reported where it is, including abnormally slow or fast rates."""
    assert abs(R.estimate(synth_csi(bpm)[1], 50.0).bpm - bpm) < 0.5


@pytest.mark.parametrize("bpm", [8, 12, 22, 30])
def test_fused_complex_rate(bpm):
    assert abs(R.estimate(synth_complex(bpm), 50.0).bpm - bpm) < 0.5


def test_noise_below_alarm_gate():
    from atheer.fusion.engine import CONF_MIN
    rng = np.random.default_rng(1)
    assert R.estimate(20 + rng.normal(0, 1, (1500, 52)), 50.0).confidence < CONF_MIN


def test_tracking_ignores_single_stray_window():
    _, amp = synth_csi(14, dur=60.0)
    est = [e.bpm for _, e in R.sliding(amp, 50.0, win_s=20, hop_s=5)]
    assert est and all(abs(b - 14) <= 0.5 for b in est)


def test_loader_keeps_only_real_subcarriers():
    from atheer.ingest import wirebreath as WB
    rng = np.random.default_rng(0)
    h = np.concatenate([rng.uniform(1000, 2000, (100, 52)), rng.uniform(0, 60, (100, 182))], 1).astype(complex)
    assert len(WB.valid_subcarriers(h)) == 52


def test_reference_rate():
    t = np.arange(0, 30, 0.02)
    assert abs(R.reference_rate(t, np.sin(2 * np.pi * 0.25 * t)) - 15) < 0.3


def test_ambiguous_half_rate_reading_is_never_confident():
    _, amp = synth_csi(14)
    ch = R.channels(amp, 50.0)
    ws = R.window_spectrum(ch, slice(None), 50.0)
    j = int(np.argmax(R._peaks_only(ws.fine)))
    bpm = R._parabolic(ws.fine, j)
    assert R._make(ws, j, 50.0, wide_bpm=bpm / 2).confidence == 0.0          # wider band sees half: maybe slow breathing
    assert R._make(ws, j, 50.0, wide_bpm=bpm).ambiguous is False
