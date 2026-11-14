"""Frequency-equivariant neural breathing-rate estimator, evaluated leave-one-subject-out on WiRe-Breath.

Idea: a breathing rate change is a time stretch, i.e. a multiplicative frequency change. On a LOG-frequency axis
that is a translation, and harmonics (2f, 3f, f/2) sit at fixed offsets. So every window becomes per-subcarrier
power spectra on a log grid (2.5-100 br/min, 40 bins/octave), processed by a small conv net with shared weights
along frequency (no position embedding), permutation-invariant pooling over subcarriers (attention+mean+max),
explicit harmonic stacking (features at f, 2f, f/2, 3f, f/3), and per-bin logits over 6-36 br/min plus one
"no breathing" logit. The net can only learn "peak vs harmonic vs interference" shapes, not absolute rates;
remaining prior leakage (spectral resolution, grid edges) is removed with time-stretch augmentation that makes
the training label distribution uniform over 6-36 br/min.

  python scripts/exp_breath_nn.py cache              # build data/cache/breath_nn_cache.npy (once)
  python scripts/exp_breath_nn.py loso [--seeds 3]   # LOSO train/eval, saves models/breath_nn_*.pt + results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.ingest import wirebreath as WB  # noqa: E402
from atheer.modules import respiration as R  # noqa: E402
from atheer.modules import breath_features as BF  # noqa: E402

CACHE = ROOT / "data" / "cache" / "breath_nn_cache.npy"
FS = 20.0
WIN, HOP_EVAL = 20.0, 5.0
N = int(WIN * FS)
WIDE = (0.03, 3.0)          # Hz, wide band so harmonics stay visible (and survive stretching)
BPO = 40                    # bins per octave
LO, HI = 2.5, 100.0         # br/min, input grid
NB = int(round(np.log2(HI / LO) * BPO)) + 1
GRID = LO * 2 ** (np.arange(NB) / BPO)               # br/min
OUT = np.where((GRID >= 6.0 - 1e-6) & (GRID <= 36.0 + 1e-6))[0]
O0, O1 = OUT[0], OUT[-1] + 1
OGRID = GRID[O0:O1]
K_SUB = 32
SIGMA_BINS = 3.0


# ----------------------------------------------------------------------------------------------- cache
def wide_filter(amp: np.ndarray, fs: float) -> np.ndarray:
    x = R.hampel(np.asarray(amp, float))
    x = signal.detrend(x, axis=0)
    return R.bandpass(x, fs, WIDE)


def to_grid(t: np.ndarray, x: np.ndarray, t0: float, n: int) -> np.ndarray:
    tu = t0 + np.arange(n) / FS
    return np.stack([np.interp(tu, t, x[:, j]) for j in range(x.shape[1])], 1)


def build_cache():
    out = []
    files = WB.find_files(ROOT / "data" / "raw" / "wirebreath" / "extracted")
    t0 = time.time()
    for k, f in enumerate(files):
        rec = WB.load(f)
        fs = rec.fs
        n20 = int((rec.time[-1] - rec.time[0]) * FS)
        x20 = to_grid(rec.time, wide_filter(rec.amp, fs), rec.time[0], n20).astype(np.float32)   # (T, 52)
        ph = np.unwrap(np.angle(rec.csi), axis=0)                                                 # sanitised phase
        p20 = to_grid(rec.time, wide_filter(ph, fs), rec.time[0], n20).astype(np.float32)
        tu, g, gfs = R.filtered_reference(rec.gt_time, rec.gt_signal)  # band-passed 0.1-0.6 Hz, whole recording
        g20 = np.interp(rec.time[0] + np.arange(n20) / FS, tu, g).astype(np.float32)
        # evaluation windows exactly as scripts/train_breath_picker.py (native-rate indices, hop 5 s)
        n, h = int(WIN * fs), int(HOP_EVAL * fs)
        xf = R.preprocess(rec.amp, fs)
        ev = []
        for end in range(n, xf.shape[0] + 1, h):
            ref = R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[end - n], rec.time[end - 1])
            if not np.isfinite(ref):
                continue
            F, grid = BF.candidate_features(xf[end - n:end], fs)
            s20 = int(round((rec.time[end - n] - rec.time[0]) * FS))
            s20 = min(s20, n20 - N)
            ev.append((s20, float(ref), float(grid[np.argmax(F[:, 0])])))
        out.append(dict(xa=x20, xp=p20, g=g20, meta=rec.meta, ev=ev))
        if k % 50 == 0:
            print(f"{k}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    np.save(CACHE, np.array(out, dtype=object), allow_pickle=True)
    print("saved", CACHE, sum(len(r["ev"]) for r in out), "eval windows")


# ----------------------------------------------------------------------------------------------- reference
_REF_NFFT = 16384


def ref_rate(g: np.ndarray) -> float:
    """Spectral peak (Hann, zero padded) of the band-passed reference segment, 6-36 br/min."""
    x = (g - g.mean()) * np.hanning(len(g))
    S = np.abs(np.fft.rfft(x, _REF_NFFT)) ** 2
    f = np.fft.rfftfreq(_REF_NFFT, 1 / FS) * 60
    m = (f >= 6) & (f <= 36)
    return float(f[m][np.argmax(S[m])])


# ----------------------------------------------------------------------------------------------- features
def make_features(torch):
    t = np.arange(N) / FS
    hann = np.hanning(N)
    E = np.exp(-2j * np.pi * np.outer(t, GRID / 60.0)) * hann[:, None]       # (N, NB)
    Er = torch.tensor(E.real, dtype=torch.float32); Ei = torch.tensor(E.imag, dtype=torch.float32)
    tt = torch.tensor((t - t.mean()) / t.std(), dtype=torch.float32)
    sel_band = torch.tensor((GRID >= 4.5) & (GRID <= 45), dtype=torch.bool)

    def feats(x, k=K_SUB):
        """x (B, N, S) filtered CSI window -> per-subcarrier log spectra (B, k, NB), pc1 spectrum (B, NB)."""
        x = x - x.mean(1, keepdim=True)
        x = x - tt[None, :, None] * (x * tt[None, :, None]).mean(1, keepdim=True)  # linear detrend
        x = x / x.std(1, keepdim=True).clamp_min(1e-9)
        xt = x.transpose(1, 2)                                                     # (B, S, N)
        P = (xt @ Er) ** 2 + (xt @ Ei) ** 2                                        # (B, S, NB)
        P = P / P.mean(2, keepdim=True).clamp_min(1e-12)
        score = P[:, :, sel_band].max(2).values                                    # peakiness, rate-agnostic
        idx = score.topk(min(k, P.shape[1]), dim=1).indices
        Ps = torch.gather(P, 1, idx[:, :, None].expand(-1, -1, NB))
        xs = torch.gather(xt, 1, idx[:, :, None].expand(-1, -1, N))                # (B, k, N)
        # first principal component across the selected subcarriers (coherence)
        u, s, _ = torch.linalg.svd(xs.transpose(1, 2), full_matrices=False)
        pc = u[:, :, 0]                                                             # (B, N)
        Pp = (pc @ Er) ** 2 + (pc @ Ei) ** 2
        Pp = Pp / Pp.mean(1, keepdim=True).clamp_min(1e-12)
        share = (s[:, 0] ** 2 / (s ** 2).sum(1).clamp_min(1e-12))
        return torch.log1p(Ps), torch.log1p(Pp), share
    return feats


def make_model(torch):
    nn = torch.nn
    F_ = torch.nn.functional

    def shift(z, d):
        """out[..., i] = z[..., i + d], zero outside."""
        if d == 0:
            return z
        out = torch.zeros_like(z)
        if d > 0:
            out[..., :-d] = z[..., d:]
        else:
            out[..., -d:] = z[..., :d]
        return out

    SH = (0, BPO, -BPO, int(round(np.log2(3) * BPO)), -int(round(np.log2(3) * BPO)))

    class Net(nn.Module):
        def __init__(self, c=16, z=20, t=32, p=0.1):
            super().__init__()
            self.enc = nn.Sequential(nn.Conv1d(1, c, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(c, c, 5, padding=4, dilation=2), nn.GELU())
            self.att = nn.Conv1d(c, 1, 1)
            self.red = nn.Conv1d(3 * c + 3, z, 1)
            self.trunk = nn.Sequential(nn.Conv1d(z * len(SH), t, 5, padding=2), nn.GELU(), nn.Dropout(p),
                                       nn.Conv1d(t, t, 5, padding=4, dilation=2), nn.GELU(), nn.Dropout(p),
                                       nn.Conv1d(t, t, 5, padding=8, dilation=4), nn.GELU())
            self.head = nn.Conv1d(t, 1, 1)
            self.null = nn.Sequential(nn.Linear(2 * t + 1, 16), nn.GELU(), nn.Linear(16, 1))

        def forward(self, Ps, Pp, share):
            B, K, L = Ps.shape
            h = self.enc(Ps.reshape(B * K, 1, L)).reshape(B, K, -1, L)                 # (B,K,c,L)
            a = torch.softmax(self.att(h.reshape(B * K, -1, L)).reshape(B, K, 1, L), dim=1)
            pooled = torch.cat([(a * h).sum(1), h.mean(1), h.max(1).values,
                                Pp[:, None], Ps.mean(1, keepdim=True), Ps.max(1).values[:, None]], 1)
            zz = self.red(pooled)
            hh = torch.cat([shift(zz, d) for d in SH], 1)
            tr = self.trunk(hh)
            logits = self.head(tr)[:, 0, O0:O1]                                         # (B, n_out)
            to = tr[:, :, O0:O1]
            g = torch.cat([to.mean(2), to.max(2).values, share[:, None]], 1)
            return torch.cat([logits, self.null(g)], 1)                                 # last = no breathing
    return Net


# ----------------------------------------------------------------------------------------------- decoding
def decode(p: np.ndarray):
    """softmax probs (n, n_out+1) -> (bpm, conf, prob on OGRID)."""
    pr = p[:, :-1]
    j = pr.argmax(1)
    bpm = np.zeros(len(p)); conf = np.zeros(len(p))
    lg = np.log2(OGRID)
    for i in range(len(p)):
        lo, hi = max(j[i] - 2, 0), min(j[i] + 3, pr.shape[1])
        w = pr[i, lo:hi]
        bpm[i] = 2 ** (np.sum(w * lg[lo:hi]) / max(w.sum(), 1e-12))
        conf[i] = pr[i, np.abs(OGRID - bpm[i]) <= 2.0].sum()
    return bpm, conf, pr


def to_bf_grid(pr: np.ndarray) -> np.ndarray:
    """probabilities on OGRID -> mass on the 0.5 br/min tracking grid of breath_features."""
    edges = np.concatenate([[BF.GRID[0] - 0.25], BF.GRID + 0.25])
    out = np.zeros((len(pr), len(BF.GRID)))
    b = np.clip(np.searchsorted(edges, OGRID) - 1, 0, len(BF.GRID) - 1)
    for k in range(len(BF.GRID)):
        out[:, k] = pr[:, b == k].sum(1)
    return out + 1e-6


# ----------------------------------------------------------------------------------------------- data sampling
class Sampler:
    def __init__(self, recs, rng, stretch=(0.5, 1.5), uniform=True):
        self.recs = recs; self.rng = rng; self.stretch = stretch; self.uniform = uniform
        self.items = []                                    # (rec idx, start) at hop 1 s, with base ref
        for i, r in enumerate(recs):
            T = r["x"].shape[0]
            for s in range(0, T - N + 1, int(FS)):
                self.items.append((i, s, ref_rate(r["g"][s:s + N])))
        self.sos = signal.butter(4, WIDE, btype="band", fs=FS, output="sos")

    def one(self, aug=True):
        rng = self.rng
        while True:
            i, s, r0 = self.items[rng.integers(len(self.items))]
            x, g = self.recs[i]["x"], self.recs[i]["g"]
            T = x.shape[0]
            if self.uniform:
                target = rng.uniform(6.0, 36.0)
                k = float(np.clip(target / r0, self.stretch[0], min(self.stretch[1], T / N)))
            else:
                k = 1.0
            m = int(round(N * k))
            if m > T:
                continue
            s2 = int(np.clip(s + (N - m) // 2, 0, T - m))
            lab = ref_rate(g[s2:s2 + m]) * k
            if not (6.0 <= lab <= 36.0):
                continue
            seg = x[s2:s2 + m]
            if m != N:
                pos = np.linspace(0, m - 1, N)
                i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, m - 1); fr = (pos - i0)[:, None]
                seg = seg[i0] * (1 - fr) + seg[i1] * fr
            seg = seg.copy()
            self.last = (i, s2, m)
            if not aug:
                return seg.astype(np.float32), lab
            # augmentation: subcarrier dropout, additive noise
            keep = rng.random(seg.shape[1]) > rng.uniform(0, 0.5)
            if keep.sum() < 8:
                keep[:] = True
            seg = seg[:, keep]
            if rng.random() < 0.5:
                seg = seg + rng.normal(0, rng.uniform(0, 1.0), seg.shape) * seg.std(0)
            return seg.astype(np.float32), lab

    def null(self):
        rng = self.rng
        S = self.recs[0]["x"].shape[1]
        z = rng.normal(size=(N + 200, S))
        a = rng.uniform(0, 0.97)
        z = signal.lfilter([1], [1, -a], z, axis=0)
        z = signal.sosfiltfilt(self.sos, z, axis=0)[100:-100]
        return z.astype(np.float32)

    def batch(self, B, null_frac=0.08):
        xs, ys = [], []
        for _ in range(B):
            if self.rng.random() < null_frac:
                xs.append(self.null()); ys.append(-1.0)
            else:
                x, y = self.one(); xs.append(x); ys.append(y)
        Smin = min(x.shape[1] for x in xs)
        xs = [x[:, self.rng.permutation(x.shape[1])[:Smin]] for x in xs]
        return np.stack(xs), np.array(ys)


def soft_targets(y: np.ndarray) -> np.ndarray:
    t = np.zeros((len(y), len(OGRID) + 1), np.float32)
    lb = np.log2(OGRID) * BPO
    for i, v in enumerate(y):
        if v < 0:
            t[i, -1] = 1
        else:
            w = np.exp(-0.5 * ((lb - np.log2(v) * BPO) / SIGMA_BINS) ** 2)
            t[i, :-1] = w / w.sum()
    return t


# ----------------------------------------------------------------------------------------------- train / predict
def eval_windows(recs):
    X, Y, C, RID = [], [], [], []
    for i, r in enumerate(recs):
        for s, ref, ctrl in r["ev"]:
            X.append(r["x"][s:s + N]); Y.append(ref); C.append(ctrl); RID.append(i)
    return np.stack(X).astype(np.float32), np.array(Y), np.array(C), np.array(RID)


def predict(torch, net, feats, X, bs=128):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Ps, Pp, sh = feats(torch.from_numpy(X[i:i + bs]))
            out.append(torch.softmax(net(Ps, Pp, sh), 1).numpy())
    return np.concatenate(out)


def train_one(torch, Net, feats, tr_recs, val, seed, epochs=40, steps=60, B=48, patience=8, log=""):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    smp = Sampler(tr_recs, rng)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    Xv, Yv = val
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train(); tl = 0
        for _ in range(steps):
            xb, yb = smp.batch(B)
            Ps, Pp, sh = feats(torch.from_numpy(xb))
            logit = net(Ps, Pp, sh)
            tgt = torch.from_numpy(soft_targets(yb))
            loss = -(tgt * torch.log_softmax(logit, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        bpm, _, _ = decode(predict(torch, net, feats, Xv))
        mae = float(np.abs(bpm - Yv).mean())
        if mae < best - 1e-3:
            best, bad = mae, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        print(f"  {log} ep{ep:02d} loss={tl / steps:.3f} valMAE={mae:.2f} best={best:.2f}", flush=True)
        if bad >= patience:
            break
    net.load_state_dict(best_state)
    return net, best


def synth_window(bpm: float | None, rng, mode="amp", fs=40.0, dur=30.0, n_breath=30, S=52):
    """Clean sinusoidal breathing in n_breath of the 52 subcarriers (amplitude and, for phase inputs, phase),
    independent white noise everywhere; same preprocessing as the real data."""
    T = int(fs * dur)
    t = np.arange(T) / fs
    chans = []
    for kind in ("amp", "phase"):
        x = rng.normal(0, 0.3, (T, S))
        if bpm is not None:
            for j in rng.choice(S, n_breath, replace=False):
                x[:, j] += rng.uniform(0.3, 1.0) * np.sign(rng.normal()) * np.sin(2 * np.pi * bpm / 60 * t + rng.uniform(0, 2 * np.pi))
        chans.append(to_grid(t, wide_filter(x, fs), 0.0, int(dur * FS)))
    x20 = chans[0] if mode == "amp" else (chans[1] if mode == "phase" else np.concatenate(chans, 1))
    s = int(5 * FS)
    return x20[s:s + N].astype(np.float32)


def stats(err):
    return {"n": int(err.size), "mae": float(err.mean()), "median": float(np.median(err)), "within2": float(np.mean(err <= 2))}


def pad52(x):
    """Some recordings have 49-51 valid subcarriers; repeat columns cyclically so windows stack (the net is
    permutation-invariant over a top-k subset, so a duplicate only re-weights an existing subcarrier)."""
    return x if x.shape[1] >= 52 else x[:, np.arange(52) % x.shape[1]]


def set_inputs(recs, mode):
    for r in recs:
        r["xa"], r["xp"] = pad52(r["xa"]), pad52(r["xp"])
        r["x"] = r["xa"] if mode == "amp" else (r["xp"] if mode == "phase" else np.concatenate([r["xa"], r["xp"]], 1))


def loso(a):
    import torch
    torch.set_num_threads(a.threads)
    feats = make_features(torch); Net = make_model(torch)
    recs = list(np.load(CACHE, allow_pickle=True))
    set_inputs(recs, a.inputs)
    subj = np.array([r["meta"]["subject"] for r in recs])
    S = sorted(set(subj))
    X, Y, C, RID = eval_windows(recs)
    WS = subj[RID]
    ENV = np.array([recs[i]["meta"]["environment"] for i in RID])
    cl = np.load(ROOT / "data" / "cache" / "breath_nn_classic.npy", allow_pickle=True)
    CL = np.array([b for row in cl for b, _, _ in row]); CLR = np.array([c for row in cl for _, _, c in row])
    assert len(CL) == len(Y) and np.allclose(CLR, Y), "classic cache misaligned"
    P = np.zeros((len(X), len(OGRID) + 1))
    rng = np.random.default_rng(123)
    # out-of-range stretched held-out windows: per test window, stretch so that the true rate is outside 10-18
    oor = {s: [] for s in S}
    synth_probs = []
    t0 = time.time()
    for si, s in enumerate(S):
        others = [q for q in S if q != s]
        vs = S[(si + 1) % len(S)] if S[(si + 1) % len(S)] != s else others[0]
        tr = [r for r, q in zip(recs, subj) if q != s and q != vs]
        vrecs = [r for r, q in zip(recs, subj) if q == vs]
        Xv, Yv, _, _ = eval_windows(vrecs)
        te = np.where(WS == s)[0]
        # stretched held-out set (built once per fold, same windows for model and control)
        ts = Sampler([r for r, q in zip(recs, subj) if q == s], np.random.default_rng(7 + si))
        sx, sy, sinfo = [], [], []
        while len(sx) < a.n_oor:
            x, y = ts.one(aug=False)
            if 10 <= y <= 18:
                continue
            sx.append(x); sy.append(y); sinfo.append(ts.last)
        # control on stretched windows: summed spectrum peak in 0.1-0.6 Hz band of the (wide-filtered) window
        ctrl_s = []
        for (ri, s2, m) in sinfo:
            xa = ts.recs[ri]["xa"][s2:s2 + m]
            pos = np.linspace(0, m - 1, N)
            xa = np.stack([np.interp(pos, np.arange(m), xa[:, j]) for j in range(xa.shape[1])], 1)
            xb = R.bandpass(xa.astype(float), FS)
            F, grid = BF.candidate_features(xb, FS)
            ctrl_s.append(grid[np.argmax(F[:, 0])])
        Smin = min(x.shape[1] for x in sx)
        sx = np.stack([x[:, :Smin] for x in sx])
        ps_fold, po_fold = [], []
        for seed in range(a.seeds):
            net, vbest = train_one(torch, Net, feats, tr, (Xv, Yv), seed=1000 * si + seed, epochs=a.epochs,
                                   steps=a.steps, log=f"{s} seed{seed}")
            torch.save(net.state_dict(), ROOT / "models" / f"breath_nn_{a.tag}_{s}_seed{seed}.pt")
            ps_fold.append(predict(torch, net, feats, X[te]))
            po_fold.append(predict(torch, net, feats, sx))
            srng = np.random.default_rng(99)
            sw = np.stack([synth_window(b, srng, a.inputs) for b in SYN] + [synth_window(None, srng, a.inputs)])
            synth_probs.append(predict(torch, net, feats, sw))
            print(f"[{s}] seed{seed} val={vbest:.2f}  elapsed {time.time() - t0:.0f}s", flush=True)
        P[te] = np.mean(ps_fold, 0)
        bo, _, _ = decode(np.mean(po_fold, 0))
        oor[s] = dict(model=np.abs(bo - np.array(sy)).tolist(), control=np.abs(np.array(ctrl_s) - np.array(sy)).tolist(),
                      true=list(map(float, sy)))
        b, _, _ = decode(P[te])
        e = np.abs(b - Y[te])
        print(f"== held-out {s}: n={te.size} MAE={e.mean():.2f} ±2={np.mean(e <= 2):.0%} | control "
              f"{np.abs(C[te] - Y[te]).mean():.2f} | OOR model {np.mean(oor[s]['model']):.2f} ctrl {np.mean(oor[s]['control']):.2f}",
              flush=True)
    bpm, conf, pr = decode(P)
    err = np.abs(bpm - Y)
    # tracking (same Viterbi as the 4.34 picker, lam 0.15) over consecutive windows of a recording
    trk = np.zeros(len(X))
    for rid in np.unique(RID):
        idx = np.where(RID == rid)[0]
        q = to_bf_grid(pr[idx]); q = q / q.sum(1, keepdims=True)
        path = BF.track(list(q), BF.GRID, 0.15)
        for i, j in zip(idx, path):
            # refine within the chosen 0.5 bin using the window's own estimate when it agrees
            trk[i] = bpm[i] if abs(bpm[i] - BF.GRID[j]) <= 0.75 else BF.GRID[j]
    errt = np.abs(trk - Y)
    edges = np.quantile(conf, np.linspace(0, 1, 6))
    qbin = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, 4)
    sp = decode(np.mean(synth_probs, 0))
    res = {
        "tag": a.tag, "seeds": a.seeds,
        "pooled": stats(err), "pooled_tracked": stats(errt), "control": stats(np.abs(C - Y)),
        "classic": stats(np.abs(CL - Y)),
        "per_env": {e: {"model": stats(err[ENV == e]), "tracked": stats(errt[ENV == e]),
                        "classic": stats(np.abs(CL - Y)[ENV == e]), "control": stats(np.abs(C - Y)[ENV == e])}
                    for e in sorted(set(ENV))},
        "per_subject": {s: {"model": stats(err[WS == s]), "tracked": stats(errt[WS == s]),
                            "control": stats(np.abs(C - Y)[WS == s]),
                            "classic": stats(np.abs(CL - Y)[WS == s])} for s in S},
        "by_conf_quintile": [{"conf_lo": float(edges[k]), "conf_hi": float(edges[k + 1]), **stats(err[qbin == k])}
                             for k in range(5)],
        "synthetic": {str(b): [float(sp[0][i]), float(sp[1][i])] for i, b in enumerate(SYN)} |
                     {"noise": [float(sp[0][-1]), float(sp[1][-1]), float(np.mean(synth_probs, 0)[-1, -1])]},
        "oor": {s: {"model_mae": float(np.mean(v["model"])), "control_mae": float(np.mean(v["control"])),
                    "model_w2": float(np.mean(np.array(v["model"]) <= 2)), "control_w2": float(np.mean(np.array(v["control"]) <= 2))}
                for s, v in oor.items()},
    }
    allm = np.concatenate([oor[s]["model"] for s in S]); allc = np.concatenate([oor[s]["control"] for s in S])
    allt = np.concatenate([oor[s]["true"] for s in S])
    res["oor_pooled"] = {"model": stats(allm), "control": stats(allc),
                         "model_slow(<10)": stats(allm[allt < 10]), "control_slow(<10)": stats(allc[allt < 10]),
                         "model_fast(>18)": stats(allm[allt > 18]), "control_fast(>18)": stats(allc[allt > 18])}
    # bias check on real held-out windows: signed error vs reference rate
    res["signed_err_by_ref"] = {f"{lo}-{hi}": float(np.mean((bpm - Y)[(Y >= lo) & (Y < hi)])) if np.any((Y >= lo) & (Y < hi)) else None
                                for lo, hi in ((6, 10), (10, 14), (14, 18), (18, 24), (24, 37))}
    np.save(ROOT / "models" / f"breath_nn_{a.tag}_loso_preds.npy", dict(P=P, Y=Y, C=C, RID=RID, WS=WS), allow_pickle=True)
    (ROOT / "models" / f"breath_nn_{a.tag}_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_subject",)}, indent=1))
    for s in S:
        d = res["per_subject"][s]
        print(f"{s}: model {d['model']['mae']:.2f}/{d['model']['within2']:.0%}  tracked {d['tracked']['mae']:.2f}/"
              f"{d['tracked']['within2']:.0%}  control {d['control']['mae']:.2f}  classic {d['classic']['mae']:.2f}")
    print(f"total {time.time() - t0:.0f}s")


SYN = (7, 8, 12, 16, 22, 30, 34)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "loso"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--n_oor", type=int, default=150)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--inputs", default="amp", choices=["amp", "phase", "both"])
    a = ap.parse_args()
    if a.cmd == "cache":
        build_cache()
    else:
        loso(a)


if __name__ == "__main__":
    main()
