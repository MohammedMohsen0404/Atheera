"""AI fusion layer: turns per-module outputs into crew state + caution & warning (C&W) alerts.

Per astronaut:
  * context state (SLEEP / REST / ACTIVE / EXERCISE) from schedule + activity + motion,
  * context-aware breathing limits (different at sleep vs exercise),
  * confidence gating — low-confidence breathing readings are shown but never alarm on their own,
  * debouncing — a condition must persist N seconds before it alarms, and alarms latch until acknowledged,
  * explainability — every alert lists the module(s), values and confidences that triggered it.
Severity mirrors ISS C&W classes: ADVISORY < CAUTION < WARNING < EMERGENCY.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

from atheer.modules.rest import MOVING_ACTIVITIES, RestTracker

SEVERITY = ["NOMINAL", "ADVISORY", "CAUTION", "WARNING", "EMERGENCY"]

# breathing limits (bpm) per context — conservative adult values, configurable per crew member
BR_LIMITS = {
    "SLEEP": (8, 22),
    "REST": (8, 25),
    "ACTIVE": (8, 32),
    "EXERCISE": (10, 40),
}
CONF_MIN = 0.45          # below this a breathing reading is "unreliable"
# motion: a 3 s roll-over in bed is normal. fall: rule chosen on validation recordings for the deployed ensemble
# (calibrated p >= FALL_PROB in one window): 16/16 held-out fall recordings, 1 false alarm in 96 (eval_activity.py)
DEBOUNCE_S = {"br_low": 15, "br_high": 20, "br_lost": 25, "motion": 6, "fall": 0, "still": 30}
FALL_PROB = 0.25  # calibrated probability, threshold chosen on validation recordings


@dataclass
class Frame:
    """One fused observation for one astronaut (1 Hz)."""
    t: float
    crew_id: str
    module: str
    occupied: bool = True
    br_available: bool = True     # False when this room has no breathing-capable sensing track
    bpm: float = math.nan
    br_conf: float = 0.0
    activity: str | None = None
    activity_probs: dict | None = None
    motion: float = 0.0
    anomaly: float = 0.0
    anomaly_reasons: list = field(default_factory=list)
    scheduled: str = "REST"       # from the crew timeline (e.g. sleep period)
    source: str = ""              # provenance: dataset file / segment


@dataclass
class Alert:
    id: int
    t: float
    crew_id: str
    module: str
    severity: str
    code: str
    title: str
    evidence: dict
    acked: bool = False
    cleared_t: float | None = None
    acked_t: float | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CrewState:
    crew_id: str
    context: str = "REST"
    status: str = "NOMINAL"
    last: Frame | None = None
    pending: dict = field(default_factory=dict)     # condition code -> first-seen time
    active: dict = field(default_factory=dict)      # condition code -> Alert
    rest: RestTracker = field(default_factory=RestTracker)   # sleep-period rest accounting


class FusionEngine:
    def __init__(self):
        self.crew: dict[str, CrewState] = {}
        self.alerts: list[Alert] = []
        self._ids = itertools.count(1)
        self.now = 0.0

    # ---------------------------------------------------------------- context
    @staticmethod
    def infer_context(f: Frame) -> str:
        act = (f.activity or "").lower()
        if act in ("run",):
            return "EXERCISE"
        if act in ("walk", "pickup", "sit_down", "stand_up") or f.anomaly > 0.5:
            return "ACTIVE"
        if f.scheduled == "SLEEP":
            return "SLEEP"
        return f.scheduled if f.scheduled in BR_LIMITS else "REST"

    # ---------------------------------------------------------------- rules
    def _conditions(self, f: Frame, ctx: str) -> dict[str, tuple[str, str, dict]]:
        """Return code -> (severity, title, evidence) for every condition currently true."""
        c: dict[str, tuple[str, str, dict]] = {}
        lo, hi = BR_LIMITS[ctx]
        br_ok = math.isfinite(f.bpm) and f.br_conf >= CONF_MIN
        ev_br = {"bpm": round(f.bpm, 1) if math.isfinite(f.bpm) else None,
                 "confidence": round(f.br_conf, 2), "limits": [lo, hi], "context": ctx,
                 "module": "respiration"}
        if f.activity_probs and f.activity_probs.get("fall", 0) >= FALL_PROB:
            c["fall"] = ("EMERGENCY", "Possible fall / uncontrolled impact",
                         {"fall_prob": round(f.activity_probs["fall"], 2), "module": "activity",
                          "reasons": f.anomaly_reasons})
        fall_open = "fall" in self.crew[f.crew_id].active if f.crew_id in self.crew else False
        if f.anomaly >= 0.8 and "fall" not in c and not fall_open and ctx != "EXERCISE":
            ev_m = {"anomaly": round(f.anomaly, 2), "module": "anomaly", "reasons": f.anomaly_reasons}
            if external_motion(f):
                # the sleeper's breathing is still tracked steadily through the motion, so it is most likely
                # someone or something nearby; if the breathing track breaks while it lasts, this escalates
                c["motion"] = ("ADVISORY", "Movement near the crew member; their breathing is still steady",
                               {**ev_m, "bpm": round(f.bpm, 1), "confidence": round(f.br_conf, 2)})
            else:
                c["motion"] = ("WARNING", "Abnormal motion pattern", ev_m)
        if br_ok and f.bpm < lo:
            c["br_low"] = ("WARNING" if f.bpm < lo - 2 else "CAUTION",
                           "Breathing rate below expected range", ev_br)
        if br_ok and f.bpm > hi and ctx in ("SLEEP", "REST"):
            c["br_high"] = ("CAUTION", "Elevated breathing rate at rest", ev_br)
        if f.br_available and f.occupied and ctx in ("SLEEP", "REST") and not br_ok and f.anomaly < 0.3:
            c["br_lost"] = ("ADVISORY", "Breathing signal unreliable — sensing degraded or position change", ev_br)
        if f.br_available and f.occupied and ctx in ("SLEEP", "REST") and f.motion < 1e-4 and not br_ok:
            c["still"] = ("WARNING", "Occupied, no motion and no breathing signal detected",
                          {**ev_br, "motion": f.motion, "module": "fusion"})
        return c

    # ---------------------------------------------------------------- main step
    def step(self, f: Frame) -> list[Alert]:
        self.now = f.t
        st = self.crew.setdefault(f.crew_id, CrewState(f.crew_id))
        st.context = self.infer_context(f)
        st.last = f
        st.rest.update(f.scheduled == "SLEEP" and f.occupied, breath_ok(f), f.anomaly, f.activity,
                       external=external_motion(f))
        new: list[Alert] = []
        conds = self._conditions(f, st.context)
        for code in list(st.pending):
            if code not in conds:
                st.pending.pop(code)
        for code, (sev, title, ev) in conds.items():
            first = st.pending.setdefault(code, f.t)
            prev = st.active.get(code)
            if prev is not None and SEVERITY.index(sev) > SEVERITY.index(prev.severity):
                prev.cleared_t = f.t          # escalation: close the old alert, raise a new, higher one
                st.active.pop(code)
            if f.t - first >= DEBOUNCE_S.get(code, 10) and code not in st.active:
                a = Alert(next(self._ids), f.t, f.crew_id, f.module, sev, code, title,
                          {**ev, "persisted_s": round(f.t - first, 1), "source": f.source})
                st.active[code] = a
                self.alerts.append(a)
                new.append(a)
        # clear conditions that have resolved (alerts stay in the log; acked+cleared leaves the board)
        for code, a in list(st.active.items()):
            if code not in conds:
                a.cleared_t = f.t
                st.active.pop(code)
        st.status = self._status(st)
        return new

    def _status(self, st: CrewState) -> str:
        """Highest severity among alerts that are unacknowledged or still physically active."""
        live = [a for a in self.alerts if a.crew_id == st.crew_id and (not a.acked or code_active(st, a))]
        return SEVERITY[max([SEVERITY.index(a.severity) for a in live] or [0])]

    def ack(self, alert_id: int) -> Alert | None:
        for a in self.alerts:
            if a.id == alert_id:
                if not a.acked:
                    a.acked, a.acked_t = True, self.now
                st = self.crew.get(a.crew_id)
                if st:
                    st.status = self._status(st)
                return a
        return None

    def snapshot(self) -> dict:
        return {
            cid: {"context": s.context, "status": s.status,
                  "active": [a.code for a in s.active.values()],
                  "rest": s.rest.summary(),
                  "last": asdict(s.last) if s.last else None}
            for cid, s in self.crew.items()
        }


def breath_ok(f: Frame) -> bool:
    return math.isfinite(f.bpm) and f.br_conf >= CONF_MIN


def external_motion(f: Frame) -> bool:
    """Motion seen during a scheduled sleep/rest period while the crew member's own breathing stays confidently tracked. A person's own body
    movement disrupts their breathing spectrum within one analysis window; motion of someone or something
    nearby usually does not (WiRe-Breath 'interference' recordings: a person moving near a still subject,
    breathing error 0.63 br/min). A heuristic, not a trained classifier: see docs/LIMITATIONS.md."""
    return f.scheduled in ("SLEEP", "REST") and breath_ok(f) and (f.activity or "").lower() not in MOVING_ACTIVITIES


def code_active(st: CrewState, a: Alert) -> bool:
    return a.code in st.active and st.active[a.code].id == a.id
