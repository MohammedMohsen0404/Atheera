"""Rest / sleep-period tracker: how much of a crew member's sleep period they spent resting, from CSI alone.

The NASA Space Apps 2026 health-monitoring challenge lists "daily sleep hours" among the indicators. Wearable
actigraphy (as in NASA's ARCHeR study) estimates sleep from wrist movement; this tracker does the same job from
Wi-Fi CSI, with nothing worn:

  * the sleep period comes from the crew schedule (like "time in bed" in a sleep diary);
  * a second of that period counts as RESTING when breathing is detected with confidence and there has been no
    body movement for the last ``SETTLE_S`` seconds (actigraphy scores sleep the same way: low movement around
    the epoch);
  * MOVING / SETTLING seconds (movement now, or within the settle time) count as awake;
  * UNCLEAR seconds (no reliable breathing and no movement) are reported separately, never counted as rest.

It estimates rest, not sleep stages: CSI cannot tell light from deep sleep, and lying still awake looks like
rest here (actigraphy has the same limit). Movement uses the per-person anomaly score (motion relative to that
person's own baseline) or an activity label that means the body is moving. Motion during which the crew
member's own breathing stays steadily tracked is attributed to someone or something nearby
(fusion.engine.external_motion) and does not interrupt rest; it is counted separately.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

SETTLE_S = 30            # quiet seconds needed after a movement before rest counts again
MOVE_SCORE = 0.5         # anomaly score above which a second counts as body movement
MOVING_ACTIVITIES = {"walk", "run", "pickup", "sit_down", "stand_up", "fall"}
STATES = ("RESTING", "SETTLING", "MOVING", "UNCLEAR", "AWAKE_PERIOD")


@dataclass
class RestTracker:
    period_s: float = 0.0        # seconds of scheduled sleep period seen so far
    rest_s: float = 0.0
    settling_s: float = 0.0
    moving_s: float = 0.0
    unclear_s: float = 0.0
    movements: int = 0           # separate movement episodes during the sleep period
    nearby_s: float = 0.0        # seconds of motion nearby while the crew member's breathing stayed steady
    state: str = "AWAKE_PERIOD"
    _since_move: float = 1e9
    _was_moving: bool = False
    _recent: deque = field(default_factory=lambda: deque(maxlen=600))   # last 10 min of states (for trends)

    def update(self, in_sleep_period: bool, breath_ok: bool, anomaly: float, activity: str | None,
               dt: float = 1.0, external: bool = False) -> str:
        """external: the motion is judged to come from someone or something nearby (the crew member's own
        breathing stays steadily tracked through it), so it does not interrupt their rest."""
        moving = (activity or "").lower() in MOVING_ACTIVITIES or (anomaly >= MOVE_SCORE and not external)
        if anomaly >= MOVE_SCORE and external and in_sleep_period:
            self.nearby_s += dt
        self._since_move = 0.0 if moving else self._since_move + dt
        if not in_sleep_period:
            self.state, self._was_moving = "AWAKE_PERIOD", moving
            return self.state
        self.period_s += dt
        if moving:
            if not self._was_moving:
                self.movements += 1
            self.state = "MOVING"; self.moving_s += dt
        elif self._since_move <= SETTLE_S:      # the first SETTLE_S quiet seconds after a movement
            self.state = "SETTLING"; self.settling_s += dt
        elif breath_ok:
            self.state = "RESTING"; self.rest_s += dt
        else:
            self.state = "UNCLEAR"; self.unclear_s += dt
        self._was_moving = moving
        self._recent.append(self.state)
        return self.state

    def summary(self) -> dict:
        p = self.period_s
        known = p - self.unclear_s
        return {
            "state": self.state,
            "sleep_period_s": round(p),
            "rest_s": round(self.rest_s),
            "awake_s": round(self.moving_s + self.settling_s),
            "unclear_s": round(self.unclear_s),
            "movements": self.movements,
            "nearby_motion_s": round(self.nearby_s),
            # share of the sleep period spent resting, and the share of the period the sensing could judge
            "rest_pct": round(100 * self.rest_s / p, 1) if p else None,
            "coverage_pct": round(100 * known / p, 1) if p else None,
            "movements_per_hour": round(3600 * self.movements / p, 1) if p >= 60 else None,
        }
