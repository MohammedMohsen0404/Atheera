"""Activity recognition — thin wrapper around the SenseFi benchmark (xyanchen/WiFi-CSI-Sensing-Benchmark, MIT).

We reuse SenseFi's UT-HAR model definitions. UT-HAR (Yousefi et al. 2017): Intel 5300, 3 antennas x 30
subcarriers = 90 features, 250-step windows.

Default model: SenseFi's ResNet18 trained with realistic augmentation (noise, receiver-gain offsets and drift,
dead subcarriers, packet loss, shifts) and per-window standardisation, on 70% of UT-HAR's recordings
(models/har_resnet18_grp_robust.pt, scripts/exp_activity_robust.py). SenseFi's published split leaks: 499 of
its 500 test windows come from recordings that also supply training windows, so its 98% is a within-recording
number. With whole recordings held out, the robust model scores 95.2% (SenseFi's recipe 91.9%) and stays at
90-95% under the perturbations above; averaging it with three GPU-trained seeds (default) gives 96.0%.
SenseFi's original weights remain available (``robust=False``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SENSEFI = ROOT / "third_party" / "WiFi-CSI-Sensing-Benchmark"
WEIGHTS = ROOT / "models" / "sensefi" / "UT-HAR"
ROBUST = ROOT / "models" / "har_resnet18_grp_robust.pt"
# ensemble of the robust model and three GPU-trained seeds (scripts/train_activity.py); averaged logits,
# temperature fitted on the grouped validation split. Chosen over the single model on validation accuracy.
ENSEMBLE = [ROBUST] + [ROOT / "models" / f"har_resnet18_grp_gpu_s{s}.pt" for s in range(3)]
ENSEMBLE_TEMPERATURE = 0.96
ROBUST_TEMPERATURE = 1.42      # single robust model, fitted on the grouped validation split
SENSEFI_TRAIN_MINMAX = (-10.666, 30.519)

# Order as documented in the SenseFi README (verified against the confusion matrix in scripts/eval_activity.py)
UT_HAR_CLASSES = ["lie_down", "fall", "walk", "pickup", "run", "sit_down", "stand_up"]
WINDOW, FEATURES = 250, 90


def _sensefi_models():
    if str(SENSEFI) not in sys.path:
        sys.path.insert(0, str(SENSEFI))
    import UT_HAR_model  # type: ignore
    return UT_HAR_model


def build(arch: str = "CNN+GRU") -> torch.nn.Module:
    m = _sensefi_models()
    table = {
        "MLP": m.UT_HAR_MLP, "LeNet": m.UT_HAR_LeNet, "ResNet18": m.UT_HAR_ResNet18,
        "ResNet50": m.UT_HAR_ResNet50, "ResNet101": m.UT_HAR_ResNet101, "RNN": m.UT_HAR_RNN,
        "GRU": m.UT_HAR_GRU, "LSTM": m.UT_HAR_LSTM, "BiLSTM": m.UT_HAR_BiLSTM,
        "CNN+GRU": m.UT_HAR_CNN_GRU, "ViT": m.UT_HAR_ViT,
    }
    return table[arch]()


def load_pretrained(arch: str = "ResNet18", path: str | Path | None = None, device: str = "cpu"):
    path = Path(path) if path else WEIGHTS / f"UT_HAR_{arch}.pt"
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):  # SenseFi may have saved the whole module
        model = obj
    else:
        model = build(arch)
        state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        model.load_state_dict(state)
    return model.to(device).eval()


def sanitize(X: np.ndarray) -> np.ndarray:
    """Repair dead subcarriers (mostly exact zeros, or frozen >10 dB away from their antenna's median) by
    interpolating across the same antenna's 30 subcarriers. Clean windows pass through unchanged."""
    X = np.array(X, np.float32, copy=True)
    xs = np.arange(30)
    for i in range(len(X)):
        x = X[i]
        cm = x.mean(0)
        far = np.abs(cm - np.repeat(np.median(cm.reshape(3, 30), 1), 30)) > 10.0
        dead = ((x == 0).mean(0) > 0.5) | ((x.std(0) < 1e-3) & far)
        if not dead.any():
            continue
        for a in range(3):
            d = dead[a * 30:(a + 1) * 30]
            if not d.any() or d.all():
                continue
            blk = x[:, a * 30:(a + 1) * 30]
            good = xs[~d]
            for r in range(blk.shape[0]):
                blk[r, d] = np.interp(xs[d], good, blk[r, good])
    return X


def norm_window(X: np.ndarray) -> np.ndarray:
    """Per-window standardisation: invariant to a global receiver-gain (dB) offset."""
    m = X.mean(axis=(1, 2), keepdims=True)
    return (X - m) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)


@dataclass
class ActivityResult:
    label: str
    probs: dict[str, float]

    @property
    def fall_prob(self) -> float:
        return self.probs.get("fall", 0.0)


class ActivityRecognizer:
    """``robust=True`` (default when its weights exist): sanitize -> per-window standardisation -> robust
    ResNet18 -> temperature-calibrated softmax. Otherwise SenseFi's pretrained weights with min-max ``norm``."""

    def __init__(self, arch: str = "ResNet18", device: str | None = None, norm: tuple[float, float] | None = None,
                 robust: bool | None = None, ensemble: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.robust = (ROBUST.exists() and arch == "ResNet18") if robust is None else robust
        self.arch = arch
        if self.robust:
            paths = [q for q in ENSEMBLE if q.exists()] if ensemble else [ROBUST]
            self.models = [load_pretrained("ResNet18", path=q, device=self.device) for q in paths]
            self.model = self.models[0]
            self.temperature = ENSEMBLE_TEMPERATURE if len(paths) == len(ENSEMBLE) else ROBUST_TEMPERATURE
            self.name = (f"ResNet18 x{len(paths)} ensemble, robust training (grouped split)" if len(paths) > 1
                         else "ResNet18, robust training (grouped split)")
        else:
            self.model = load_pretrained(arch, device=self.device)
            self.models = [self.model]
            self.temperature = 1.0
            self.name = f"SenseFi {arch} (pretrained, unchanged)"
        # SenseFi min-max normalises each split with its global min/max; we store those values
        self.norm = norm

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        x = np.asarray(x, np.float32)
        if x.ndim == 2:
            x = x[None]
        x = sanitize(x)
        if self.robust:
            x = norm_window(x)
        elif self.norm is not None:
            lo, hi = self.norm
            x = (x - lo) / (hi - lo)
        return torch.from_numpy(np.ascontiguousarray(x.reshape(len(x), 1, WINDOW, FEATURES))).to(self.device)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> list[ActivityResult]:
        xt = self._prep(x)
        out = torch.stack([m(xt).float() for m in self.models]).mean(0)
        if self.arch == "CNN+GRU" and not self.robust:
            out = torch.log(out.clamp_min(1e-8))  # SenseFi's CNN+GRU already ends in Softmax: don't apply it twice
        p = torch.softmax(out / self.temperature, -1).cpu().numpy()
        return [ActivityResult(UT_HAR_CLASSES[int(r.argmax())],
                               {c: float(v) for c, v in zip(UT_HAR_CLASSES, r)}) for r in p]
