"""Crew-status brief for the flight surgeon / CAPCOM.

Uses Claude to turn recent fused state + alerts into a short, plain-language brief.
Always has a deterministic template fallback (no key, offline, timeout, refusal), so the demo never blocks.
The LLM only *summarises* engine output — it never makes the alert decisions (those are rule-based, auditable).
"""
from __future__ import annotations

import json
import os

MODEL = "claude-opus-5-5"
SYSTEM = (
    "You write crew health & safety briefs for a space-station flight surgeon from a contactless "
    "Wi-Fi sensing system. Input is JSON: crew states and recent alerts with evidence and confidence. "
    "Write at most 5 short bullet lines, most urgent first: who, where, what was detected, how confident, "
    "and the suggested next step (e.g. 'call crew member on comm loop'). Use only facts in the input; "
    "say when a reading is low-confidence. Never diagnose a medical condition. Plain text, no markdown headers."
)


def template_brief(snapshot: dict, alerts: list[dict]) -> str:
    order = ["EMERGENCY", "WARNING", "CAUTION", "ADVISORY", "NOMINAL"]
    lines = []
    for a in sorted(alerts, key=lambda a: (order.index(a["severity"]), -a["t"]))[:4]:
        ev = a.get("evidence", {})
        conf = ev.get("confidence", ev.get("fall_prob", ev.get("anomaly")))
        state = "acknowledged" if a.get("acked") else "UNACKNOWLEDGED"
        lines.append(f"- [{a['severity']}] {a['crew_id']} ({a['module']}): {a['title']}"
                     f" — confidence {conf}, {state}. Suggest: contact crew member on comm loop.")
    nominal = [c for c, s in snapshot.items() if s["status"] == "NOMINAL"]
    if nominal:
        lines.append(f"- Nominal: {', '.join(nominal)}.")
    return "\n".join(lines) or "- All crew nominal."


def llm_brief(snapshot: dict, alerts: list[dict], timeout: float = 20.0) -> tuple[str, str]:
    """Return (text, source) where source is 'claude' or 'template'."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return template_brief(snapshot, alerts), "template"
    try:
        import anthropic
    except ImportError:
        return template_brief(snapshot, alerts), "template"
    compact = {
        "crew": {c: {k: s[k] for k in ("context", "status", "active")} |
                 {"module": (s.get("last") or {}).get("module"),
                  "bpm": (s.get("last") or {}).get("bpm"),
                  "br_conf": (s.get("last") or {}).get("br_conf"),
                  "activity": (s.get("last") or {}).get("activity"),
                  "sleep_period_rest": s.get("rest")}
                 for c, s in snapshot.items()},
        "recent_alerts": alerts[-12:],
    }
    try:
        client = anthropic.Anthropic(timeout=timeout, max_retries=1)
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=1024,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(compact, default=str)}],
        )
        if resp.stop_reason == "refusal":
            return template_brief(snapshot, alerts), "template"
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return (text, "claude") if text else (template_brief(snapshot, alerts), "template")
    except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.RateLimitError,
            anthropic.APIStatusError):
        return template_brief(snapshot, alerts), "template"
