"""Fair baselines for exp_breath_nn.py on the same 1403 evaluation windows (corrected WB.load):
classic, summed-spectrum control, and the control + the same Viterbi tracking (lam 0.15) the NN uses.
Also signed error by reference rate for each method. Usage: python scripts/breath_nn_baselines.py <tag>"""
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.ingest import wirebreath as WB  # noqa: E402
from atheer.modules import respiration as R  # noqa: E402
from atheer.modules import breath_features as BF  # noqa: E402

cache = ROOT / "data" / "cache" / "breath_nn_ctrlspec.npy"
if not cache.exists():
    rows = []
    for k, f in enumerate(WB.find_files(ROOT / "data" / "raw" / "wirebreath" / "extracted")):
        rec = WB.load(f); fs = rec.fs
        n, h = int(20 * fs), int(5 * fs)
        xf = R.preprocess(rec.amp, fs)
        for end in range(n, xf.shape[0] + 1, h):
            ref = R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[end - n], rec.time[end - 1])
            if not np.isfinite(ref):
                continue
            F, grid = BF.candidate_features(xf[end - n:end], fs)
            rows.append((k, F[:, 0].astype(np.float32), ref))
    np.save(cache, np.array(rows, dtype=object), allow_pickle=True)
rows = np.load(cache, allow_pickle=True)
rid = np.array([r[0] for r in rows]); ssum = np.stack([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
ctrl = BF.GRID[ssum.argmax(1)]
trk = np.zeros(len(Y))
for r in np.unique(rid):
    idx = np.where(rid == r)[0]
    q = np.clip(ssum[idx], 1e-6, None); q = q / q.sum(1, keepdims=True)
    for i, j in zip(idx, BF.track(list(q), BF.GRID, 0.15)):
        trk[i] = BF.GRID[j]
tag = sys.argv[1] if len(sys.argv) > 1 else "amp"
d = np.load(ROOT / "models" / f"breath_nn_{tag}_loso_preds.npy", allow_pickle=True).item()
assert np.allclose(d["Y"], Y)
cl = np.load(ROOT / "data" / "cache" / "breath_nn_classic.npy", allow_pickle=True)
CL = np.array([b for row in cl for b, _, _ in row])
recs_meta = [r["meta"] for r in np.load(ROOT / "data" / "cache" / "breath_nn_cache.npy", allow_pickle=True)]
env = np.array([recs_meta[i]["environment"] for i in d["RID"]])
out = {}
for name, p in (("classic", CL), ("control", ctrl), ("control_tracked", trk)):
    e = np.abs(p - Y)
    out[name] = {"all": [float(e.mean()), float((e <= 2).mean())],
                 **{v: [float(e[env == v].mean()), float((e[env == v] <= 2).mean())] for v in ("Quiet", "Interference")},
                 **{s: [float(e[d["WS"] == s].mean()), float((e[d["WS"] == s] <= 2).mean())] for s in np.unique(d["WS"])},
                 "signed": {f"{lo}-{hi}": float(np.mean((p - Y)[(Y >= lo) & (Y < hi)])) for lo, hi in
                            ((6, 10), (10, 14), (14, 18), (18, 24), (24, 37))}}
print(json.dumps(out, indent=1))
