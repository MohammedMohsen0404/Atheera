"""Shared helpers for scripts/exp_activity_robust.py (UT-HAR robustness / honest-split study).

Everything here works on RAW UT-HAR windows (N, 250, 90) as stored by SenseFi. Values range ~-10..30 and behave like
amplitude in dB (Intel 5300 CSI tool), so a receiver gain / AGC change is modelled as an additive dB offset.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.modules.activity import UT_HAR_CLASSES, build, load_pretrained  # noqa: E402

DATA = ROOT / "data" / "raw" / "sensefi" / "UT_HAR"
FALL = UT_HAR_CLASSES.index("fall")
T, D = 250, 90


# ----------------------------------------------------------------------------------------------- data
def load_split(name: str):
    X = np.load(open(DATA / "data" / f"X_{name}.csv", "rb")).reshape(-1, T, D).astype(np.float32)
    y = np.load(open(DATA / "label" / f"y_{name}.csv", "rb")).astype(int)
    return X, y


def load_all():
    parts = {s: load_split(s) for s in ("train", "val", "test")}
    X = np.concatenate([parts[s][0] for s in parts])
    y = np.concatenate([parts[s][1] for s in parts])
    split = np.concatenate([[s] * len(parts[s][1]) for s in parts])
    return X, y, split


def recording_groups(X: np.ndarray, min_shared: int = 10):
    """Recover the original continuous recordings: SenseFi windows were cut with a sliding window (stride 100 rows),
    so windows of one recording share exact CSI rows. Union-find over windows sharing >= min_shared rows.
    Returns (group id per window, row offset of each window inside its recording, pair-overlap histogram)."""
    N = len(X)
    rows = defaultdict(list)
    for i in range(N):
        for r in range(T):
            rows[X[i, r].tobytes()].append((i, r))
    pair = defaultdict(list)  # (a,b) -> list of (ra - rb)
    for occ in rows.values():
        if 1 < len(occ) < 20:
            for a in range(len(occ)):
                for b in range(a + 1, len(occ)):
                    (i, ri), (j, rj) = occ[a], occ[b]
                    if i != j:
                        k = (i, j) if i < j else (j, i)
                        pair[k].append(ri - rj if i < j else rj - ri)
    par = list(range(N))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    adj = defaultdict(list)
    for (i, j), offs in pair.items():
        if len(offs) >= min_shared:
            par[find(i)] = find(j)
            off = Counter(offs).most_common(1)[0][0]  # pos_j = pos_i + off
            adj[i].append((j, off))
            adj[j].append((i, -off))
    g = np.array([find(i) for i in range(N)])
    pos = np.full(N, np.nan)
    for i in range(N):
        if np.isnan(pos[i]):
            pos[i] = 0
            stack = [i]
            while stack:
                a = stack.pop()
                for b, off in adj[a]:
                    if np.isnan(pos[b]):
                        pos[b] = pos[a] + off
                        stack.append(b)
    for gid in np.unique(g):  # shift so each recording starts at 0
        m = g == gid
        pos[m] -= pos[m].min()
    _, g = np.unique(g, return_inverse=True)
    hist = Counter(len(v) for v in pair.values())
    return g, pos.astype(int), dict(sorted(hist.items(), key=lambda kv: -kv[1])[:8])


def grouped_split(y, g, frac=(0.7, 0.1, 0.2), seed=0):
    """Assign whole recordings to train/val/test, stratified by activity class."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), dtype=object)
    for c in np.unique(y):
        gids = np.unique(g[y == c])
        rng.shuffle(gids)
        n = len(gids)
        n_tr, n_va = int(round(frac[0] * n)), int(round(frac[1] * n))
        for k, gid in enumerate(gids):
            out[g == gid] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    return out.astype(str)


# --------------------------------------------------------------------------------------- normalisation
def norm_minmax(X, lo, hi):
    return (X - lo) / (hi - lo)


def norm_window(X):
    """Per-window standardisation (one mean/std per 250x90 window). Invariant to global dB offset."""
    m = X.mean(axis=(1, 2), keepdims=True)
    s = X.std(axis=(1, 2), keepdims=True) + 1e-6
    return (X - m) / s


# ---------------------------------------------------------------------------------------- perturbations
def _interp_rows(x, keep):
    """Linear interpolation of dropped rows (packet loss) for one window x (T,D); keep is bool (T,)."""
    idx = np.arange(len(keep))
    if keep.sum() < 2:
        return np.repeat(x[keep][:1], len(keep), 0) if keep.any() else x
    prev = np.maximum.accumulate(np.where(keep, idx, -1))
    nxt = np.minimum.accumulate(np.where(keep, idx, len(keep))[::-1])[::-1]
    prev2 = np.where(prev < 0, nxt, prev)
    nxt2 = np.where(nxt >= len(keep), prev2, nxt)
    w = np.where(nxt2 == prev2, 0.0, (idx - prev2) / np.maximum(nxt2 - prev2, 1))[:, None]
    return (1 - w) * x[prev2] + w * x[nxt2]


def _shift(x, s):
    """Shift window in time by s rows, edge rows replicated (window misaligned with the activity)."""
    if s == 0:
        return x
    if s > 0:
        return np.concatenate([np.repeat(x[:1], s, 0), x[:-s]])
    return np.concatenate([x[-s:], np.repeat(x[-1:], -s, 0)])


def _stretch(x, f):
    """Time-scale by factor f (f<1: activity done faster -> crop of f*T rows resampled to T), centred."""
    L = int(round(T * f))
    src = np.linspace(0, L - 1, T)
    if L <= T:
        st = (T - L) // 2
        seg = x[st:st + L]
    else:  # slower activity: pad by edge replication then sample
        pad = (L - T) // 2
        seg = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], L - T - pad, 0)])
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, L - 1)
    w = (src - i0)[:, None]
    return (1 - w) * seg[i0] + w * seg[i1]


def _dyn_std(x):
    return float(np.mean(x.std(0)))  # temporal std per subcarrier, averaged (= "motion signal" level)


def perturb(X, kind: str, level, seed: int = 0):
    """Apply one perturbation to raw windows. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    N = len(X)
    if kind == "clean":
        return X
    if kind == "noise_snr_db":  # white noise relative to the window's own temporal (dynamic) std
        for i in range(N):
            X[i] += rng.normal(0, _dyn_std(X[i]) * 10 ** (-level / 20), X[i].shape)
        return X
    if kind == "gain_offset_db":  # global receiver gain change, random sign per window
        return X + (rng.choice([-1, 1], N) * level)[:, None, None]
    if kind == "agc_per_antenna_db":  # independent gain per RX chain (3 antennas x 30 subcarriers)
        off = rng.uniform(-level, level, (N, 3))
        return X + np.repeat(off, 30, axis=1)[:, None, :]
    if kind == "gain_drift_db":  # linear AGC drift across the window, random sign
        ramp = np.linspace(-0.5, 0.5, T)[None, :, None] * (rng.choice([-1, 1], N) * level)[:, None, None]
        return X + ramp
    if kind == "subcarrier_flatline":  # fraction of subcarriers frozen at their window mean (dead / no update)
        k = int(round(level * D))
        for i in range(N):
            c = rng.choice(D, k, replace=False)
            X[i][:, c] = X[i][:, c].mean(0)
        return X
    if kind == "subcarrier_zero":  # fraction of subcarriers reported as 0 (driver glitch)
        k = int(round(level * D))
        for i in range(N):
            X[i][:, rng.choice(D, k, replace=False)] = 0.0
        return X
    if kind == "time_shift":  # +-level rows, random sign, edge-padded
        for i in range(N):
            X[i] = _shift(X[i], int(rng.choice([-1, 1]) * level))
        return X
    if kind == "time_stretch":
        for i in range(N):
            X[i] = _stretch(X[i], level)
        return X
    if kind == "packet_loss":  # i.i.d. row drops, linearly interpolated
        for i in range(N):
            X[i] = _interp_rows(X[i], rng.random(T) >= level)
        return X
    if kind == "packet_loss_burst":  # one contiguous outage of `level` rows, interpolated
        for i in range(N):
            keep = np.ones(T, bool)
            st = rng.integers(0, T - level)
            keep[st:st + level] = False
            X[i] = _interp_rows(X[i], keep)
        return X
    raise ValueError(kind)


PERTURBATIONS = [("clean", 0)] + \
    [("noise_snr_db", v) for v in (20, 10, 5, 0)] + \
    [("gain_offset_db", v) for v in (3, 6)] + \
    [("agc_per_antenna_db", v) for v in (2, 4)] + \
    [("gain_drift_db", v) for v in (3, 6)] + \
    [("subcarrier_flatline", v) for v in (0.1, 0.3)] + \
    [("subcarrier_zero", 0.1)] + \
    [("time_shift", v) for v in (25, 50)] + \
    [("time_stretch", v) for v in (0.8, 1.2)] + \
    [("packet_loss", v) for v in (0.1, 0.3, 0.5)] + \
    [("packet_loss_burst", 50)]


def augment(X, rng):
    """Training augmentation: random subset of the realistic perturbations, sampled per window (raw dB domain)."""
    X = X.copy()
    for i in range(len(X)):
        x = X[i]
        if rng.random() < 0.5:
            x = x + rng.normal(0, _dyn_std(x) * 10 ** (-rng.uniform(3, 30) / 20), x.shape)
        if rng.random() < 0.5:
            x = x + rng.uniform(-6, 6)
        if rng.random() < 0.4:
            x = x + np.repeat(rng.uniform(-3, 3, 3), 30)[None, :]
        if rng.random() < 0.3:
            x = x + np.linspace(-0.5, 0.5, T)[:, None] * rng.uniform(-6, 6)
        if rng.random() < 0.3:
            c = rng.choice(D, rng.integers(1, 19), replace=False)
            x[:, c] = x[:, c].mean(0) if rng.random() < 0.7 else 0.0
        if rng.random() < 0.4:
            x = _interp_rows(x, rng.random(T) >= rng.uniform(0.05, 0.5))
        if rng.random() < 0.4:
            x = _shift(x, int(rng.integers(-50, 51)))
        if rng.random() < 0.3:
            x = _stretch(x, rng.uniform(0.8, 1.2))
        X[i] = x
    return X.astype(np.float32)


# ------------------------------------------------------------------------------------------ inference
def model_logits(model, Xn, arch: str, bs: int = 64, device="cpu"):
    """Logits for normalised windows. SenseFi's CNN+GRU ends in nn.Softmax, so its output is already a
    probability vector; we return log-probs for it (applying softmax again would squash probs to <=0.31)."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xn), bs):
            o = model(torch.from_numpy(np.ascontiguousarray(Xn[i:i + bs], np.float32)).view(-1, 1, T, D).to(device))
            out.append(o.float().cpu())
    o = torch.cat(out)
    if arch == "CNN+GRU":
        o = torch.log(o.clamp_min(1e-8))
    return o.numpy()


def softmax(z, temp=1.0):
    z = z / temp
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y)
    best = (1.0, 1e9)
    for T_ in np.exp(np.linspace(np.log(0.05), np.log(20), 400)):
        nll = F.cross_entropy(z / T_, t).item()
        if nll < best[1]:
            best = (float(T_), nll)
    return best[0]


def ece(P, y, bins=15):
    conf = P.max(1)
    pred = P.argmax(1)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def metrics(P, y, thr=0.6):
    pred = P.argmax(1)
    pf = P[:, FALL]
    isf = y == FALL
    alert = pf >= thr
    from sklearn.metrics import roc_auc_score
    return {"acc": round(float((pred == y).mean()), 4),
            "fall_recall": round(float(alert[isf].mean()), 4),
            "fall_false_alarms": int(alert[~isf].sum()),
            "fall_fa_per_100": round(float(100 * alert[~isf].mean()), 2),
            "fall_auc": round(float(roc_auc_score(isf, pf)), 4)}


# --------------------------------------------------------------------------------------- motion features
def motion_profile(X):
    """Per-row motion energy: mean |first difference| across subcarriers, lightly smoothed. (N, T-1)."""
    e = np.abs(np.diff(X, axis=1)).mean(2)
    k = np.ones(9) / 9
    return np.stack([np.convolve(r, k, mode="same") for r in e])


def motion_features(X):
    e = motion_profile(X)
    q = e.shape[1] // 4
    peak = e.max(1)
    base = np.median(e, 1) + 1e-6
    burst = peak / base                      # how impulsive the strongest movement is
    tail = e[:, -q:].mean(1) / (peak + 1e-6)  # stillness after the event (low = still)
    head = e[:, :q].mean(1) / (peak + 1e-6)
    return {"burst": burst, "tail": tail, "head": head, "peak": peak}


# ------------------------------------------------------------------------------------------- training
def train_model(arch, Xtr, ytr, Xva, yva, normfn, epochs, lr, aug, init=None, seed=0, bs=64, log=print,
                label_smoothing=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = load_pretrained(arch, path=init, device="cpu") if init else build(arch)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xva_n = normfn(Xva)
    best = (-1, None, -1)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tl, n = 0.0, 0
        for b in range(len(Xtr) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            xb = Xtr[idx]
            if aug:
                xb = augment(xb, rng)
            xb = torch.from_numpy(normfn(xb).astype(np.float32)).view(-1, 1, T, D)
            out = model(xb)
            if arch == "CNN+GRU":
                loss = F.nll_loss(torch.log(out.clamp_min(1e-8)), torch.from_numpy(ytr[idx]))
            else:
                loss = F.cross_entropy(out, torch.from_numpy(ytr[idx]), label_smoothing=label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
            n += len(idx)
        va_logits = model_logits(model, Xva_n, arch)
        va_acc = float((va_logits.argmax(1) == yva).mean())
        va_nll = nll(softmax(va_logits), yva)
        log(f"  ep {ep + 1:2d}/{epochs} loss {tl / n:.4f} val_acc {va_acc:.4f} val_nll {va_nll:.4f}")
        score = va_acc - 1e-3 * va_nll  # accuracy, ties broken by NLL
        if score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    model.eval()
    return model, best[2]


# ------------------------------------------------------------------------------ test-time input hygiene
def sanitize(X):
    """Repair dead subcarriers before the model sees them: a column that is mostly exact zeros, or frozen
    (std < 1e-3) more than 10 dB away from its antenna's median level, is replaced, row by row, by linear interpolation across the valid
    subcarriers of the same antenna (3 antennas x 30 subcarriers). Returns repaired copy."""
    X = X.copy()
    xs = np.arange(30)
    for i in range(len(X)):
        x = X[i]
        cm = x.mean(0)
        far = np.abs(cm - np.repeat(np.median(cm.reshape(3, 30), 1), 30)) > 10.0
        # zero-reported columns, or frozen columns at an implausible level; a column frozen near its normal
        # level is left alone (measured: interpolating those costs accuracy, the model tolerates them)
        dead = ((x == 0).mean(0) > 0.5) | ((x.std(0) < 1e-3) & far)
        if not dead.any():
            continue
        for a in range(3):
            d = dead[a * 30:(a + 1) * 30]
            if not d.any() or d.all():
                continue
            blk = x[:, a * 30:(a + 1) * 30]
            good = xs[~d]
            for r in range(T):
                blk[r, d] = np.interp(xs[d], good, blk[r, good])
            x[:, a * 30:(a + 1) * 30] = blk
        X[i] = x
    return X


def align_offset(X, mu):
    """Remove a global receiver-gain change for min-max models: shift each window so its mean equals the
    training-set mean mu (dB domain), then apply the usual fixed min-max."""
    return X - X.mean(axis=(1, 2), keepdims=True) + mu
