"""Scenario replay: streams precomputed dataset tracks through anomaly detection + fusion at 1 Hz (x speed).

A scenario (YAML) assigns each crew member a habitat module and a timeline of segments; each segment
plays one track (real dataset recording or, in dev, a synthetic placeholder). Every frame carries the
track's provenance so the dashboard can show exactly which public recording is being replayed.
"""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict
from pathlib import Path

import yaml

from atheer.fusion.engine import FusionEngine, Frame
from atheer.modules.anomaly import AnomalyDetector

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "results" / "tracks"


def load_track(track_id: str, prefer_real: bool = True) -> dict:
    """Load a track; 'auto:<kind>:<hint>' resolves to a real track if one exists, else the synthetic one."""
    p = TRACKS / f"{track_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    raise FileNotFoundError(f"track {track_id} not found in {TRACKS}")


def resolve(seg: dict) -> str:
    """Segments list candidate track ids in order; the first that exists wins (real before synthetic)."""
    cands = seg["track"] if isinstance(seg["track"], list) else [seg["track"]]
    for c in cands:
        if (TRACKS / f"{c}.json").exists():
            return c
    raise FileNotFoundError(f"none of {cands} exist in {TRACKS}")


class ScenarioPlayer:
    def __init__(self, scenario_path: str | Path):
        self.path = Path(scenario_path)
        self.spec = yaml.safe_load(self.path.read_text())
        self.duration = float(self.spec.get("duration_s", 600))
        self.speed = float(self.spec.get("speed", 1.0))
        self.reset()

    def reset(self):
        self.spec = yaml.safe_load(self.path.read_text())  # drops injected segments
        self.t = 0.0
        self.running = False
        self.fusion = FusionEngine()
        self.detectors = {c["id"]: AnomalyDetector() for c in self.spec["crew"]}
        self._cache: dict[str, dict] = {}
        self._kind: dict[str, tuple] = {}
        self.events_fired: set[int] = set()

    def _track(self, tid: str) -> dict:
        if tid not in self._cache:
            self._cache[tid] = load_track(tid)
        return self._cache[tid]

    def _segment(self, crew: dict, t: float) -> dict:
        segs = sorted(crew["timeline"], key=lambda s: s["at"])
        cur = segs[0]
        for s in segs:
            if s["at"] <= t:
                cur = s
        return cur

    def frame_for(self, crew: dict, t: float) -> tuple[Frame, dict]:
        seg = self._segment(crew, t)
        module = seg.get("module", crew["module"])
        tid = resolve(seg)
        tr = self._track(tid)
        n = len(tr["t"])
        i = int(t - seg["at"] + seg.get("offset", 0))
        i = i % n if seg.get("loop", True) else min(i, n - 1)
        f = Frame(t=t, crew_id=crew["id"], module=module, scheduled=seg.get("scheduled", "REST"),
                  occupied=seg.get("occupied", True),
                  source=f"{tr['dataset']} :: {tr['id']} @ {i}s")
        extra = {"track": tid, "synthetic": tr.get("synthetic", False), "wave": None, "gt_bpm": None,
                 "csi": tr["csi"][i] if tr.get("csi") and i < len(tr["csi"]) else None,
                 "injected": bool(seg.get("injected"))}
        if tr["kind"] == "breath":
            b = tr["bpm"][i]
            f.bpm = math.nan if b is None else b
            f.br_conf = tr["conf"][i] or 0.0
            f.motion = tr["motion"][i] or 0.0
            hz = tr.get("wave_hz", 10)
            extra["wave"] = tr["wave"][i * hz:(i + 1) * hz]
            extra["gt_bpm"] = tr["gt_bpm"][i]
            f.activity = "lie_down" if f.scheduled == "SLEEP" else None
        else:
            probs = dict(zip(tr["meta"].get("classes") or
                             ["lie_down", "fall", "walk", "pickup", "run", "sit_down", "stand_up"], tr["probs"][i]))
            f.activity, f.activity_probs, f.motion = tr["activity"][i], probs, tr["motion"][i]
            f.br_available = seg.get("bpm") is not None
            if seg.get("bpm") is not None:  # breathing for this crew comes from a co-scheduled breath track
                f.bpm, f.br_conf = seg["bpm"], seg.get("br_conf", 0.0)
        det = self.detectors[crew["id"]]
        # a different data source (dataset/hardware, or synthetic vs recorded) means a different motion scale:
        # relearn the personal baseline instead of flagging the switch itself as abnormal motion
        src = (tr["kind"], bool(tr.get("synthetic")), tr.get("dataset"))
        if self._kind.get(crew["id"]) not in (None, src):
            det.hist.clear()
        self._kind[crew["id"]] = src
        res = det.update(f.motion, (f.activity_probs or {}).get("fall", 0.0), f.bpm, f.br_conf)
        f.anomaly, f.anomaly_reasons = res.score, res.reasons
        return f, extra

    # ---------------------------------------------------------------- live what-if injection
    INJECT = {
        # kind: (segment template, human label)
        "fall": ({"scheduled": "ACTIVE", "track": ["uthar_fall", "syn_fall"], "loop": False, "hold_s": 6,
                  "then": {"scheduled": "REST", "track": ["uthar_lie_down", "syn_lie_down"]}}, "Fall injected"),
        "breath_slow": ({"scheduled": "SLEEP", "track": ["wbx_s2_slowing", "syn_sleep_slowing"]}, "Breathing slows"),
        "interference": ({"scheduled": "REST", "track": ["wb_s3_sitting_interference", "syn_rest_noisy"]},
                         "Radio interference"),
        "exercise": ({"scheduled": "EXERCISE", "track": ["uthar_run", "syn_run"]}, "Exercise session"),
    }

    def inject(self, crew_id: str, kind: str) -> dict:
        crew = next((c for c in self.spec["crew"] if c["id"] == crew_id), None)
        if crew is None:
            raise KeyError(crew_id)
        t0 = float(self.t)
        if kind == "recover":
            base = sorted(crew["_base"] if "_base" in crew else crew["timeline"], key=lambda s: s["at"])
            first = dict(base[0]); first.update(at=t0, injected=True, label="Back to schedule")
            crew["timeline"] = [s for s in crew["timeline"] if s["at"] < t0] + [first]
            return first
        tmpl, label = self.INJECT[kind]
        crew.setdefault("_base", [dict(s) for s in crew["timeline"]])
        seg = {k: v for k, v in tmpl.items() if k not in ("hold_s", "then")}
        seg.update(at=t0, injected=True, label=label)
        new = [seg]
        if "then" in tmpl:
            new.append({**tmpl["then"], "at": t0 + tmpl["hold_s"], "injected": True, "label": "Lying still"})
        # injected segments replace whatever was scheduled from now on
        crew["timeline"] = [s for s in crew["timeline"] if s["at"] < t0] + new
        return seg

    def timeline(self) -> list[dict]:
        out = []
        for c in self.spec["crew"]:
            for s in c["timeline"]:
                if s.get("label"):
                    out.append({"at": s["at"], "crew": c["id"], "label": s["label"], "injected": bool(s.get("injected"))})
        return sorted(out, key=lambda e: e["at"])

    def tick(self) -> dict:
        frames, new_alerts = [], []
        for crew in self.spec["crew"]:
            f, extra = self.frame_for(crew, self.t)
            new_alerts += [a.to_dict() for a in self.fusion.step(f)]
            d = asdict(f)
            d["bpm"] = None if not math.isfinite(f.bpm) else round(f.bpm, 2)
            d.update(extra)
            d["context"] = self.fusion.crew[f.crew_id].context
            d["status"] = self.fusion.crew[f.crew_id].status
            d["rest"] = self.fusion.crew[f.crew_id].rest.summary()
            d["name"] = crew.get("name", crew["id"])
            frames.append(d)
        # scripted crew acknowledgements etc.
        for k, ev in enumerate(self.spec.get("events", [])):
            if k not in self.events_fired and self.t >= ev["at"] and ev.get("type") == "auto_ack":
                for a in self.fusion.alerts:
                    if a.crew_id == ev["crew"] and not a.acked:
                        self.fusion.ack(a.id)
                self.events_fired.add(k)
        msg = {"type": "telemetry", "t": self.t, "duration": self.duration, "frames": frames,
               "new_alerts": new_alerts}
        self.t += 1.0
        if self.t >= self.duration:
            self.running = False
        return msg

    async def run(self, emit):
        self.running = True
        while self.running:
            await emit(self.tick())
            await asyncio.sleep(1.0 / max(self.speed, 0.1))
