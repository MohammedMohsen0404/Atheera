"""FastAPI backend: scenario control, live telemetry over WebSocket, alert acknowledgement, AI brief.

  uvicorn atheer.api.main:app --reload --port 8010
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from atheer.fusion.brief import llm_brief
from atheer.replay.engine import ScenarioPlayer

ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "scenarios" / "iss_night_shift.yaml"
RESULTS = ROOT / "results"

app = FastAPI(title="Atheer — Spatial-CSI Astronaut Health & Safety Guard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.player = ScenarioPlayer(SCENARIO)
        self.task: asyncio.Task | None = None
        self.last: dict | None = None

    async def broadcast(self, msg: dict):
        if msg.get("type") == "telemetry":
            self.last = msg
        data = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                self.clients.discard(ws)

    def state(self) -> dict:
        p = self.player
        return {"type": "state", "running": p.running, "t": p.t, "duration": p.duration, "speed": p.speed,
                "scenario": p.spec.get("name"), "modules": p.spec.get("modules", []),
                "crew": [{"id": c["id"], "name": c.get("name", c["id"]), "module": c["module"]} for c in p.spec["crew"]],
                "alerts": [a.to_dict() for a in p.fusion.alerts], "timeline": p.timeline(),
                "inject_kinds": list(p.INJECT) + ["recover"]}


hub = Hub()


@app.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_text(json.dumps(hub.state(), default=str))
    if hub.last:
        await ws.send_text(json.dumps(hub.last, default=str))
    try:
        while True:
            await ws.receive_text()  # keepalive; control goes through REST
    except WebSocketDisconnect:
        hub.clients.discard(ws)


@app.get("/api/state")
def get_state():
    return hub.state()


@app.post("/api/scenario/start")
async def start():
    if hub.player.t >= hub.player.duration:
        hub.player.reset()
    await stop_task()
    hub.task = asyncio.create_task(hub.player.run(hub.broadcast))
    await asyncio.sleep(0)  # let run() mark the player as running before we report state
    await hub.broadcast(hub.state())
    return {"ok": True}


async def stop_task():
    hub.player.running = False
    if hub.task and not hub.task.done():
        hub.task.cancel()
        try:
            await hub.task
        except asyncio.CancelledError:
            pass
    hub.task = None


@app.post("/api/scenario/pause")
async def pause():
    await stop_task()
    await hub.broadcast(hub.state())
    return {"ok": True}


@app.post("/api/scenario/reset")
async def reset():
    await stop_task()
    hub.player.reset()
    hub.last = None
    await hub.broadcast(hub.state())
    return {"ok": True}


class Seek(BaseModel):
    t: float


@app.post("/api/scenario/seek")
async def seek(body: Seek):
    """Jump to time t by replaying the script from 0 (fusion state is rebuilt; injected events are dropped)."""
    was_running = hub.player.running
    await stop_task()
    hub.player.reset()
    target = max(0.0, min(body.t, hub.player.duration - 1))
    msg = None
    while hub.player.t < target:
        msg = hub.player.tick()
    if msg:
        hub.last = msg
    await hub.broadcast(hub.state())
    if msg:
        await hub.broadcast({**msg, "seek": True})
    if was_running:
        hub.task = asyncio.create_task(hub.player.run(hub.broadcast))
        await asyncio.sleep(0)
        await hub.broadcast(hub.state())
    return {"ok": True, "t": hub.player.t}


class Speed(BaseModel):
    speed: float


@app.post("/api/scenario/speed")
async def speed(s: Speed):
    hub.player.speed = max(0.25, min(s.speed, 32))
    await hub.broadcast(hub.state())
    return {"ok": True, "speed": hub.player.speed}


class Inject(BaseModel):
    crew_id: str
    kind: str


@app.post("/api/inject")
async def inject(body: Inject):
    if body.kind not in list(hub.player.INJECT) + ["recover"]:
        raise HTTPException(400, f"unknown event '{body.kind}'")
    try:
        seg = hub.player.inject(body.crew_id, body.kind)
    except KeyError:
        raise HTTPException(404, f"no crew member '{body.crew_id}'")
    await hub.broadcast(hub.state())
    return {"ok": True, "segment": seg}


@app.post("/api/alerts/{alert_id}/ack")
async def ack(alert_id: int):
    a = hub.player.fusion.ack(alert_id)
    if not a:
        raise HTTPException(404, "alert not found")
    await hub.broadcast({"type": "ack", "alert": a.to_dict()})
    return a.to_dict()


@app.get("/api/brief")
async def brief():
    snap = hub.player.fusion.snapshot()
    alerts = [a.to_dict() for a in hub.player.fusion.alerts]
    text, source = await asyncio.to_thread(llm_brief, snap, alerts)
    return {"t": hub.player.t, "text": text, "source": source}


@app.get("/api/evidence")
def evidence():
    """Offline evaluation results (written by scripts/eval_*.py) for the 'Model evidence' tab."""
    out = {}
    for p in sorted(RESULTS.glob("eval_*.json")):
        if p.stem.endswith("_baseline"):
            continue
        out[p.stem] = json.loads(p.read_text())
    # accuracies published by SenseFi for the same pretrained weights. A copy is kept in results/ (tracked in git),
    # so a fresh clone and the Docker image show them too; the file next to the downloaded weights is the fallback.
    published = RESULTS / "sensefi_published.json"
    rep = ROOT / "models" / "sensefi" / "UT-HAR" / "UT-HAR_result"
    if published.exists():
        out["sensefi_reported"] = json.loads(published.read_text(encoding="utf-8"))["accuracy"]
    elif rep.exists():
        out["sensefi_reported"] = {k.strip(): float(v.strip().rstrip("%")) / 100
                                   for k, v in (ln.split(":") for ln in rep.read_text().splitlines() if ":" in ln)}
    return out


dist = ROOT / "dashboard" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
