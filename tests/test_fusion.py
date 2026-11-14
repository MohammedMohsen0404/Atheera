import math
from atheer.fusion.engine import FusionEngine, Frame


def run(frames):
    e = FusionEngine(); out = []
    for f in frames: out += e.step(f)
    return e, out


def test_nominal_sleep_no_alerts():
    e, out = run([Frame(t, "A", "CQ", bpm=14, br_conf=0.9, motion=0.01, scheduled="SLEEP") for t in range(120)])
    assert out == [] and e.crew["A"].status == "NOMINAL" and e.crew["A"].context == "SLEEP"


def test_fall_is_immediate_emergency():
    fr = [Frame(t, "B", "LAB", bpm=16, br_conf=0.8, motion=0.02) for t in range(10)]
    fr.append(Frame(10, "B", "LAB", activity="fall", activity_probs={"fall": 0.9}, anomaly=0.95, motion=0.3))
    e, out = run(fr)
    assert len(out) == 1 and out[0].severity == "EMERGENCY" and out[0].t == 10
    assert e.crew["B"].status == "EMERGENCY"
    e.ack(out[0].id)
    e.step(Frame(11, "B", "LAB", bpm=16, br_conf=0.8, motion=0.02))
    assert e.crew["B"].status == "NOMINAL"


def test_fall_probability_below_threshold_is_not_an_alert():
    from atheer.fusion.engine import FALL_PROB
    fr = [Frame(t, "B", "LAB", bpm=16, br_conf=0.8, motion=0.02, activity_probs={"fall": FALL_PROB * 0.8}) for t in range(20)]
    _, out = run(fr)
    assert not any(a.code == "fall" for a in out)


def test_low_br_debounced_and_confidence_gated():
    # low-confidence low readings never alarm
    _, out = run([Frame(t, "C", "CQ", bpm=5, br_conf=0.2, motion=0.01, scheduled="SLEEP") for t in range(60)])
    assert all(a.code != "br_low" for a in out)
    # confident low reading alarms only after debounce (15 s)
    _, out = run([Frame(t, "C", "CQ", bpm=5, br_conf=0.9, motion=0.01, scheduled="SLEEP") for t in range(30)])
    lows = [a for a in out if a.code == "br_low"]
    assert len(lows) == 1 and lows[0].t == 15 and lows[0].severity == "WARNING"


def test_exercise_context_raises_limit():
    _, out = run([Frame(t, "D", "NODE1", bpm=34, br_conf=0.9, activity="run", motion=0.2) for t in range(60)])
    assert not any(a.code == "br_high" for a in out)


def test_escalation_caution_to_warning():
    fr = [Frame(t, "E", "CQ", bpm=7.5, br_conf=0.9, scheduled="SLEEP") for t in range(20)]
    fr += [Frame(t, "E", "CQ", bpm=5.0, br_conf=0.9, scheduled="SLEEP") for t in range(20, 40)]
    e, out = run(fr)
    assert [a.severity for a in out if a.code == "br_low"] == ["CAUTION", "WARNING"]


def test_no_br_lost_without_breathing_sensor():
    _, out = run([Frame(t, "F", "LAB", br_available=False, activity="lie_down", scheduled="REST") for t in range(60)])
    assert out == []


def test_brief_motion_burst_is_not_an_alert_but_sustained_is():
    base = [Frame(t, "D", "CQ", bpm=14, br_conf=0.9, motion=0.01, scheduled="SLEEP") for t in range(30)]
    burst = [Frame(30 + t, "D", "CQ", bpm=14, br_conf=0.9, motion=0.2, anomaly=0.95, scheduled="SLEEP") for t in range(3)]
    after = [Frame(33 + t, "D", "CQ", bpm=14, br_conf=0.9, motion=0.01, scheduled="SLEEP") for t in range(20)]
    _, out = run(base + burst + after)
    assert not any(a.code == "motion" for a in out)
    sustained = [Frame(30 + t, "D", "CQ", bpm=14, br_conf=0.9, motion=0.2, anomaly=0.95, scheduled="SLEEP") for t in range(12)]
    _, out = run(base + sustained)
    assert any(a.code == "motion" for a in out)
