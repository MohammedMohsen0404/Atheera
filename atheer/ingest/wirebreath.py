"""WiRe-Breath loader — Python port of third_party/WiRe-Breath/load_wire_breath_data.m.

Dataset: Ge, Wang, Ho, IEEE JBHI 2025 (doi 10.1109/JBHI.2025.3619664), CC-BY-4.0.
Each file: ~30 s of CSI at 50 Hz, 234 subcarriers, plus a reference respiration signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

REQUIRED = ("csi_complex", "csi_abs", "time", "gt_time", "gt_signal", "metadata")
POSTURES = ("Foetus", "Freefaller", "Log", "Normal", "Semi", "Sitting", "Soldier", "Starfish", "Yearner")


@dataclass
class BreathRecording:
    path: Path
    csi: np.ndarray          # complex, (T, S)
    time: np.ndarray         # (T,) seconds
    gt_time: np.ndarray      # (G,)
    gt_signal: np.ndarray    # (G,)
    fs: float
    meta: dict = field(default_factory=dict)

    @property
    def amp(self) -> np.ndarray:
        return np.abs(self.csi)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


def _mat_struct_to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    out = {}
    if hasattr(obj, "_fieldnames"):
        for k in obj._fieldnames:
            out[k] = getattr(obj, k)
    elif isinstance(obj, np.ndarray) and obj.dtype.names:
        for k in obj.dtype.names:
            v = obj[k]
            while isinstance(v, np.ndarray) and v.size == 1 and v.dtype == object:
                v = v.item()
            out[k] = v.item() if isinstance(v, np.ndarray) and v.size == 1 else v
    return out


def _load_raw(path: Path) -> dict:
    try:
        return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:  # MATLAB v7.3 -> HDF5
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            for k in REQUIRED:
                if k == "metadata" or k not in f:
                    continue
                d = f[k][()]
                if d.dtype.names and {"real", "imag"} <= set(d.dtype.names):
                    d = d["real"] + 1j * d["imag"]
                out[k] = np.asarray(d).T.squeeze()
            out["metadata"] = {}
        return out


def parse_filename(path: Path) -> dict:
    """Pull posture / environment / room / subject hints from the file path."""
    s = str(path)
    meta = {"file": path.name}
    for p in POSTURES:
        if re.search(rf"(?i)(^|[_\-/\\]){p}([_\-.]|$)", s):
            meta["posture"] = p
            break
    m = re.search(r"(?i)(quiet|interference)", s)
    if m:
        meta["environment"] = m.group(1).capitalize()
    m = re.search(r"(?i)room\s*_?(\d+)", s)
    if m:
        meta["room"] = f"Room{m.group(1)}"
    m = re.search(r"(?i)(?:subject|sub|participant|p)[_\-]?(\d+)", path.stem)
    if m:
        meta["subject"] = f"S{int(m.group(1))}"
    return meta


def valid_subcarriers(csi: np.ndarray, rel: float = 0.1) -> np.ndarray:
    """Indices of subcarriers that carry a channel. WiRe-Breath stores 234 columns but only positions 0-52
    hold signal (median |H| ~1,400) and position 25 is a null; the rest are near-zero quantisation junk."""
    med = np.median(np.abs(csi), axis=0)
    return np.where(med >= rel * med.max())[0]


def sanitize_phase(csi: np.ndarray, pos: np.ndarray | None = None, passes: int = 2) -> np.ndarray:
    """Remove each packet's random phase offset (CFO) and slope (packet-detection delay / STO).

    Single-antenna CSI phase jumps by ~1-2 rad between packets. Relative to a reference packet, the per-packet
    error is a + b*k over subcarrier position k: b is estimated from amplitude-weighted adjacent-pair phase
    differences (robust to gaps such as the null), a from the de-sloped mean, in two passes (reference = first
    packet, then the pass-1 mean). Median packet-to-packet step drops from ~1.6 rad to ~0.05-0.1 rad."""
    k = np.arange(csi.shape[1], dtype=float) if pos is None else np.asarray(pos, float)
    ref = csi[0]
    out = csi
    for _ in range(passes):
        q = csi * np.conj(ref)[None, :]
        gap = np.diff(k)
        pair = q[:, 1:] * np.conj(q[:, :-1])
        b = np.angle((np.abs(pair) * np.exp(1j * np.angle(pair) / gap[None, :])).sum(1))
        a = np.angle((q * np.exp(-1j * b[:, None] * k[None, :])).sum(1))
        out = csi * np.exp(-1j * (a[:, None] + b[:, None] * k[None, :]))
        ref = out.mean(0)
    return out


def _resample(csi: np.ndarray, t: np.ndarray, tu: np.ndarray) -> np.ndarray:
    """Resample complex CSI to the times ``tu``. Interpolating raw complex values would blend packets with
    unrelated random phases and corrupt the amplitude (~20% too low on WiRe-Breath), so the amplitude is
    interpolated on its own and the phase only after per-packet sanitisation."""
    amp = np.abs(csi)
    cs = sanitize_phase(csi)  # caller passes valid subcarriers only
    out = np.empty((tu.size, csi.shape[1]), complex)
    for k in range(csi.shape[1]):
        a = np.interp(tu, t, amp[:, k])
        z = np.interp(tu, t, cs[:, k].real) + 1j * np.interp(tu, t, cs[:, k].imag)
        out[:, k] = a * np.exp(1j * np.angle(z))
    return out


def load(path: str | Path) -> BreathRecording:
    path = Path(path)
    raw = _load_raw(path)
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise KeyError(f"Missing variables {missing} in {path}")
    csi = np.asarray(raw["csi_complex"])
    if csi.ndim == 1:
        csi = csi[:, None]
    t = np.asarray(raw["time"], dtype=float).ravel()
    if csi.shape[0] != t.size and csi.shape[1] == t.size:
        csi = csi.T
    if csi.shape[0] != t.size:
        raise ValueError(f"CSI rows {csi.shape[0]} != timestamps {t.size} in {path}")
    gt_t = np.asarray(raw["gt_time"], dtype=float).ravel()
    gt_s = np.asarray(raw["gt_signal"], dtype=float).ravel()
    if gt_t.size != gt_s.size:
        raise ValueError(f"gt_time/gt_signal length mismatch in {path}")
    # CSI packets arrive irregularly (a nominal 50 Hz file can span ~37 s), so resample onto a uniform grid
    # at the recording's true mean rate using its timestamps; the reference signal is already in real time.
    order = np.argsort(t)
    t, csi = t[order], csi[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, csi = t[keep], csi[keep]
    fs = (t.size - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 50.0
    tu = t[0] + np.arange(t.size) / fs
    csi = csi[:, valid_subcarriers(csi)]  # drop the null and the ~180 empty columns
    csi = _resample(csi, t, tu)
    t = tu
    meta = parse_filename(path)
    md = _mat_struct_to_dict(raw["metadata"])
    for k, v in md.items():
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            meta[k] = v.item() if hasattr(v, "item") else v
        elif isinstance(v, np.ndarray) and v.dtype.kind in "U":
            meta[k] = str(v)
    return BreathRecording(path, csi.astype(np.complex64), t, gt_t, gt_s, float(fs), meta)


def find_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.mat"))
