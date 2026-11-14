"""Post-hoc check (single pre-chosen rule, no tuning): NN probabilities (held-out LOSO) x the non-learned
summed spectrum, geometric mean on the 0.5 br/min grid, then the same Viterbi tracking (lam 0.15).
Usage: python scripts/breath_nn_fusion.py <tag>"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp_breath_nn as M  # noqa: E402
from atheer.modules import breath_features as BF  # noqa: E402

tag = sys.argv[1]
d = np.load(ROOT / "models" / f"breath_nn_{tag}_loso_preds.npy", allow_pickle=True).item()
rows = np.load(ROOT / "data" / "cache" / "breath_nn_ctrlspec.npy", allow_pickle=True)
ssum = np.stack([r[1] for r in rows]); Y = d["Y"]; RID = d["RID"]
meta = [r["meta"] for r in np.load(M.CACHE, allow_pickle=True)]
env = np.array([meta[i]["environment"] for i in RID])
bpm, conf, pr = M.decode(d["P"])
q_nn = M.to_bf_grid(pr); q_nn /= q_nn.sum(1, keepdims=True)
q_c = np.clip(ssum, 1e-6, None); q_c /= q_c.sum(1, keepdims=True)
for name, q in (("nn", q_nn), ("fused", np.sqrt(q_nn * q_c))):
    q = q / q.sum(1, keepdims=True)
    raw = BF.GRID[q.argmax(1)]
    raw = np.where(np.abs(raw - bpm) <= 0.75, bpm, raw)
    trk = np.zeros(len(Y))
    for r in np.unique(RID):
        idx = np.where(RID == r)[0]
        for i, j in zip(idx, BF.track(list(q[idx]), BF.GRID, 0.15)):
            trk[i] = bpm[i] if abs(bpm[i] - BF.GRID[j]) <= 0.75 else BF.GRID[j]
    for lab, p in (("raw", raw), ("tracked", trk)):
        e = np.abs(p - Y)
        print(f"{tag} {name:5s} {lab:7s} all {e.mean():.3f}/{(e<=2).mean():.1%}  Quiet {e[env=='Quiet'].mean():.3f}"
              f"  Interf {e[env=='Interference'].mean():.3f}/{(e[env=='Interference']<=2).mean():.1%}  "
              + " ".join(f"{s}:{e[d['WS']==s].mean():.2f}" for s in np.unique(d['WS'])))
