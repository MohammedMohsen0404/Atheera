import math

import numpy as np

from atheer.fusion.engine import FusionEngine, Frame
from atheer.modules.anomaly import motion_index
from atheer.modules.rest import SETTLE_S, RestTracker


def run(tr, n, **kw):
    for _ in range(n):
        tr.update(**{"in_sleep_period": True, "breath_ok": True, "anomaly": 0.0, "activity": None, **kw})


def test_still_with_breathing_is_rest():
    tr = RestTracker(); run(tr, 100)
    s = tr.summary()
    assert s["rest_s"] == 100 and s["awake_s"] == 0 and s["movements"] == 0 and s["rest_pct"] == 100.0


def test_movement_then_settling_counts_awake():
    tr = RestTracker(); run(tr, 50); run(tr, 5, anomaly=0.9); run(tr, 60)
    s = tr.summary()
    assert s["movements"] == 1
    assert s["awake_s"] == 5 + SETTLE_S          # 5 s moving + settle time before rest counts again
    assert s["rest_s"] == 50 + 60 - SETTLE_S


def test_no_breathing_is_unclear_not_rest():
    tr = RestTracker(); run(tr, 40, breath_ok=False)
    s = tr.summary()
    assert s["rest_s"] == 0 and s["unclear_s"] == 40 and s["coverage_pct"] == 0.0


def test_outside_sleep_period_not_counted():
    tr = RestTracker(); run(tr, 30, in_sleep_period=False)
    assert tr.summary()["sleep_period_s"] == 0 and tr.state == "AWAKE_PERIOD"


def test_external_motion_does_not_interrupt_rest():
    tr = RestTracker(); run(tr, 30); run(tr, 10, anomaly=0.9, external=True); run(tr, 10)
    s = tr.summary()
    assert s["movements"] == 0 and s["rest_s"] == 50 and s["nearby_motion_s"] == 10


def test_moving_activity_always_counts():
    tr = RestTracker(); run(tr, 3, activity="walk", external=True)
    assert tr.summary()["movements"] == 1 and tr.state == "MOVING"


def _frame(t, anomaly, bpm, conf):
    return Frame(t=t, crew_id="A", module="CQ", scheduled="SLEEP", bpm=bpm, br_conf=conf,
                 anomaly=anomaly, motion=0.01, activity="lie_down")


def test_nearby_motion_advisory_escalates_when_breathing_lost():
    fe = FusionEngine(); alerts = []
    for t in range(10):                       # motion, but the sleeper's breathing is tracked steadily
        alerts += fe.step(_frame(t, 0.95, 14.0, 0.9))
    assert [a.severity for a in alerts] == ["ADVISORY"]
    for t in range(10, 14):                   # motion continues and the breathing track is lost
        alerts += fe.step(_frame(t, 0.95, math.nan, 0.0))
    assert alerts[-1].code == "motion" and alerts[-1].severity == "WARNING"
    assert fe.snapshot()["A"]["rest"]["movements"] == 1


def test_gain_step_is_not_motion():
    rng = np.random.default_rng(0); fs = 50
    shape = rng.uniform(5, 15, 52)
    amp = shape * (1 + 0.002 * rng.normal(size=(150, 52)))
    amp[75:] *= 2.0                            # receiver gain (AGC) step: every subcarrier scales at once
    assert motion_index(amp, fs, gain_norm=True) < 0.2 * motion_index(amp, fs)
    moving = amp.copy(); moving[75:] *= rng.uniform(0.5, 1.5, 52)   # body motion changes the pattern
    assert motion_index(moving, fs, gain_norm=True) > 3 * motion_index(amp, fs, gain_norm=True)
