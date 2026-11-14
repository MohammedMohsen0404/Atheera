"""GPU training of the robust activity model on UT-HAR's grouped split (whole recordings held out).

Same recipe as scripts/exp_activity_robust.py (per-window standardisation + realistic augmentation + dead-subcarrier
repair at test time), with a longer schedule and several seeds. Model selection (best epoch), temperature and the
choice single-vs-ensemble use the validation recordings only; the test recordings are reported once at the end.

  python scripts/train_activity.py --epochs 80 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import har_common as H  # noqa: E402
from atheer.modules.activity import build, norm_window, sanitize  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def logits(model, X, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(norm_window(sanitize(X[i:i + bs])), np.float32)).view(-1, 1, 250, 90)
            out.append(model(xb.to(DEV)).float().cpu())
    return torch.cat(out).numpy()


def train(Xtr, ytr, Xva, yva, epochs, lr, seed, bs=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = build("ResNet18").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    best = (-1.0, None, 0)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(Xtr)); tl = 0.0
        for b in range(len(Xtr) // bs):
            idx = perm[b * bs:(b + 1) * bs]
            xb = torch.from_numpy(norm_window(H.augment(Xtr[idx], rng)).astype(np.float32)).view(-1, 1, 250, 90).to(DEV)
            loss = F.cross_entropy(model(xb), torch.from_numpy(ytr[idx]).to(DEV), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        lv = logits(model, Xva)
        acc = float((lv.argmax(1) == yva).mean()); score = acc - 1e-3 * H.nll(H.softmax(lv), yva)
        if score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  seed {seed} ep {ep + 1:3d}/{epochs} loss {tl / (len(Xtr) // bs):.4f} val_acc {acc:.4f}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    return model, best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    X, y, _ = H.load_all()
    grp = np.load(ROOT / "models" / "har_groups.npz"); gs = grp["gsplit"]
    tr, va, te = gs == "train", gs == "val", gs == "test"
    print(f"device {DEV}; train {tr.sum()} / val {va.sum()} / test {te.sum()} windows")
    res, LV, LT = {}, [], []
    for s in a.seeds:
        t0 = time.time()
        model, ep = train(X[tr], y[tr], X[va], y[va], a.epochs, a.lr, s)
        path = ROOT / "models" / f"har_resnet18_grp_gpu_s{s}.pt"
        torch.save(model.state_dict(), path)
        lv, lt = logits(model, X[va]), logits(model, X[te]); LV.append(lv); LT.append(lt)
        res[f"seed{s}"] = {"best_epoch": ep, "val_acc": float((lv.argmax(1) == y[va]).mean()),
                           "test_acc": float((lt.argmax(1) == y[te]).mean()), "minutes": round((time.time() - t0) / 60, 1),
                           "path": str(path.relative_to(ROOT))}
        print(res[f"seed{s}"], flush=True)
    ens_v = np.mean(LV, 0); ens_t = np.mean(LT, 0)
    res["ensemble"] = {"val_acc": float((ens_v.argmax(1) == y[va]).mean()), "test_acc": float((ens_t.argmax(1) == y[te]).mean()),
                       "temperature": float(H.fit_temperature(ens_v, y[va]))}
    best_single = max((k for k in res if k.startswith("seed")), key=lambda k: res[k]["val_acc"])
    res["choice_on_val"] = "ensemble" if res["ensemble"]["val_acc"] >= res[best_single]["val_acc"] else best_single
    res["epochs"], res["lr"], res["device"] = a.epochs, a.lr, DEV
    print(json.dumps(res, indent=1))
    (ROOT / "results" / "train_activity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
