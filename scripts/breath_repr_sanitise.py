"""CSI subcarrier validity + per-packet phase sanitisation for WiRe-Breath (helper for exp_breath_repr.py).

Finding: of the 234 reported subcarriers only the first 53 positions carry signal (|H| ~ 700-2200); position 25
is a null (|H| ~ 50) and positions 53..233 are near-zero quantisation junk (|H| ~ 3-70). So the usable channel is
52 contiguous subcarriers (positions 0-24, 26-52), i.e. a 20 MHz-wide channel, with a linear phase model in the
position index.

Raw phase carries a random per-packet common offset a(t) (CFO) and slope b(t) (packet-detection / sampling
time offset). Removed per packet, BEFORE any time-domain filtering or resampling:
  q_k(t)  = H_k(t) conj(Href_k)                      (static channel shape removed)
  b(t)    = angle( sum_k q_{k+1} conj(q_k) )         (circular slope, amplitude weighted, adjacent valid pairs)
  a(t)    = angle( sum_k q_k exp(-j b k) )
  Hs_k(t) = H_k(t) exp(-j (a + b k))
Two passes: Href = packet 0, then Href = mean of the first-pass result.
"""
from __future__ import annotations

import numpy as np

N_POS = 53       # positions 0..52 hold the real channel
NULL_POS = (25,)


def valid_positions(c: np.ndarray, thr: float = 0.2) -> np.ndarray:
    """Positions among 0..52 whose median |H| is at least `thr` x the median of that block (drops the null)."""
    m = np.median(np.abs(c[:, :N_POS]), 0)
    return np.where(m >= thr * np.median(m))[0]


def _pass(c, ref, k):
    q = c * np.conj(ref)[None, :]
    d = np.diff(k)
    pairs = q[:, 1:] * np.conj(q[:, :-1])
    step = np.exp(1j * np.angle(pairs) / d[None, :]) * np.abs(pairs)
    b = np.angle(step.sum(1))
    a = np.angle((q * np.exp(-1j * b[:, None] * k[None, :])).sum(1))
    return c * np.exp(-1j * (a[:, None] + b[:, None] * k[None, :])), a, b


def sanitise(c: np.ndarray, pos: np.ndarray | None = None):
    """c: raw complex CSI (T, 234) -> (sanitised H at the valid positions (T, P), positions, a, b)."""
    c = np.asarray(c, complex)
    pos = valid_positions(c) if pos is None else pos
    h = c[:, pos]
    k = pos.astype(float)
    s, _, _ = _pass(h, h[0], k)
    s, a, b = _pass(h, s.mean(0), k)
    return s, pos, a, b
