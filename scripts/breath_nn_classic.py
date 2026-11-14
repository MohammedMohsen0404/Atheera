"""Classic amplitude estimator (R.estimate_classic) on exactly the evaluation windows of exp_breath_nn.py
(same file order and native-rate window indices). Writes data/cache/breath_nn_classic.npy (list per file)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atheer.ingest import wirebreath as WB  # noqa: E402
from atheer.modules import respiration as R  # noqa: E402

out = []
for f in WB.find_files(ROOT / "data" / "raw" / "wirebreath" / "extracted"):
    rec = WB.load(f); fs = rec.fs
    n, h = int(20 * fs), int(5 * fs)
    xf = R.preprocess(rec.amp, fs)
    row = []
    for end in range(n, xf.shape[0] + 1, h):
        ref = R.reference_rate(rec.gt_time, rec.gt_signal, rec.time[end - n], rec.time[end - 1])
        if not np.isfinite(ref):
            continue
        c = R.estimate_classic(xf[end - n:end], fs)
        row.append((c.bpm, c.confidence, ref))
    out.append(row)
np.save(ROOT / "data" / "cache" / "breath_nn_classic.npy", np.array(out, dtype=object), allow_pickle=True)
e = np.array([abs(a - c) for r in out for a, b, c in r]); print("classic MAE", e.mean(), (e <= 2).mean(), e.size)
