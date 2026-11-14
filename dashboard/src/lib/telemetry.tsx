"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { SEVERITY_ORDER, type Alert, type CrewDef, type Frame, type StateMsg, type TelemetryMsg } from "./types";

const HISTORY_S = 180;
const FLUSH_MS = 200;
export const CSI_ROWS = 120; // 12 s at 10 Hz
export const CSI_SUB = 16;

export interface CrewHistory {
  t: number[];
  bpm: (number | null)[];
  gt: (number | null)[];
  conf: number[];
  wave: number[]; // last 12 s at 10 Hz
  csi: Uint8Array[]; // rows of 16 subcarriers, last 12 s at 10 Hz
}

const decodeCsi = (hex: string): Uint8Array[] => {
  const rows: Uint8Array[] = [];
  for (let r = 0; r < hex.length / (CSI_SUB * 2); r++) {
    const row = new Uint8Array(CSI_SUB);
    for (let c = 0; c < CSI_SUB; c++) row[c] = parseInt(hex.substr((r * CSI_SUB + c) * 2, 2), 16);
    rows.push(row);
  }
  return rows;
};

/** In `next dev` the page runs on :3000 and the backend on :8010; behind FastAPI both share one origin. */
function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const host = location.port === "3000" ? `${location.hostname}:8010` : location.host;
  return `${proto}://${host}/ws/telemetry`;
}

export async function post<T = unknown>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`the server answered ${r.status}`);
  return r.json();
}

interface Telemetry {
  state: StateMsg | null;
  frames: Record<string, Frame>;
  framesRef: React.RefObject<Record<string, Frame>>;
  alerts: Alert[];
  /** The newest warning or emergency, held for a few seconds so the Crew page can announce it once. */
  notice: Alert | null;
  t: number;
  connected: boolean;
  running: boolean;
  crew: CrewDef[];
  modules: Record<string, string>;
  history: React.RefObject<Record<string, CrewHistory>>;
  lastFrameAt: React.RefObject<number>;
  speedRef: React.RefObject<number>;
  focus: string | null;
  setSelected: (id: string | null) => void;
  selected: string | null;
  togglePlay: () => void;
  ack: (id: number) => Promise<unknown>;
}

const Ctx = createContext<Telemetry | null>(null);

export function useTelemetry() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTelemetry must be used inside <TelemetryProvider>");
  return v;
}

/** How long a new warning or emergency stays announced, in real seconds. */
const NOTICE_MS = 6000;

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<StateMsg | null>(null);
  const [frames, setFrames] = useState<Record<string, Frame>>({});
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [t, setT] = useState(0);
  const [connected, setConnected] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const history = useRef<Record<string, CrewHistory>>({});
  const framesRef = useRef<Record<string, Frame>>({});
  const lastFrameAt = useRef<number>(0);
  const speedRef = useRef(4);
  const [notice, setNotice] = useState<Alert | null>(null);
  const noticeTimer = useRef<number | undefined>(undefined);
  // Telemetry can arrive 16 times a second at high replay speed. Canvases read the refs directly;
  // React state (and so the whole page) updates at most every FLUSH_MS, which keeps scrolling smooth.
  const latestT = useRef(0);
  const flushTimer = useRef<number | undefined>(undefined);
  const lastFlush = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    const connect = () => {
      ws = new WebSocket(wsUrl());
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "state") {
          const s = msg as StateMsg;
          setState(s);
          speedRef.current = s.speed;
          setAlerts(s.alerts);
          setT(s.t);
          if (s.t === 0) {
            history.current = {};
            framesRef.current = {};
            setFrames({});
            window.clearTimeout(noticeTimer.current);
            setNotice(null);
          }
        } else if (msg.type === "telemetry") {
          const m = msg as TelemetryMsg & { seek?: boolean };
          if (m.seek) history.current = {};
          lastFrameAt.current = performance.now();
          latestT.current = m.t;
          const next: Record<string, Frame> = {};
          for (const f of m.frames) {
            next[f.crew_id] = f;
            const h = (history.current[f.crew_id] ??= { t: [], bpm: [], gt: [], conf: [], wave: [], csi: [] });
            h.t.push(f.t);
            h.bpm.push(f.bpm);
            h.gt.push(f.gt_bpm);
            h.conf.push(f.br_conf);
            if (h.t.length > HISTORY_S) {
              h.t.shift(); h.bpm.shift(); h.gt.shift(); h.conf.shift();
            }
            if (f.wave) {
              h.wave.push(...f.wave);
              if (h.wave.length > CSI_ROWS) h.wave.splice(0, h.wave.length - CSI_ROWS);
            }
            if (f.csi) {
              h.csi.push(...decodeCsi(f.csi));
              if (h.csi.length > CSI_ROWS) h.csi.splice(0, h.csi.length - CSI_ROWS);
            }
          }
          framesRef.current = next;
          if (flushTimer.current === undefined) {
            const wait = Math.max(0, FLUSH_MS - (performance.now() - lastFlush.current));
            flushTimer.current = window.setTimeout(() => {
              flushTimer.current = undefined;
              lastFlush.current = performance.now();
              setFrames(framesRef.current);
              setT(latestT.current);
            }, wait);
          }
          if (m.new_alerts.length) {
            setAlerts((a) => [...a, ...m.new_alerts]);
            // One announcement, no pop-ups: the most serious new warning or emergency is held for a few seconds.
            const urgent = m.new_alerts
              .filter((a) => a.severity === "EMERGENCY" || a.severity === "WARNING")
              .sort((a, b) => SEVERITY_ORDER.indexOf(b.severity) - SEVERITY_ORDER.indexOf(a.severity))[0];
            if (urgent) {
              setNotice(urgent);
              window.clearTimeout(noticeTimer.current);
              noticeTimer.current = window.setTimeout(() => setNotice(null), NOTICE_MS);
            }
          }
        } else if (msg.type === "ack") {
          setAlerts((a) => a.map((x) => (x.id === msg.alert.id ? msg.alert : x)));
          setNotice((n) => (n && n.id === msg.alert.id ? null : n));
        }
      };
    };
    connect();
    return () => {
      window.clearTimeout(retry);
      window.clearTimeout(flushTimer.current);
      window.clearTimeout(noticeTimer.current);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  const running = state?.running ?? false;
  const togglePlay = useCallback(() => {
    if (!connected) return;
    post(running ? "/api/scenario/pause" : "/api/scenario/start").catch((e) => toast.error(`The replay didn't respond: ${e.message}`));
  }, [connected, running]);
  const ack = useCallback((id: number) => post(`/api/alerts/${id}/ack`), []);

  const crew = useMemo(() => state?.crew ?? [], [state?.crew]);
  const modules = useMemo(() => Object.fromEntries((state?.modules ?? []).map((m) => [m.id, m.name])), [state?.modules]);
  const focus = selected ?? crew[0]?.id ?? null;

  const value: Telemetry = {
    state, frames, framesRef, alerts, notice, t, connected, running, crew, modules, history, lastFrameAt, speedRef,
    focus, selected, setSelected, togglePlay, ack,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Current sample of the 10 Hz breathing waveform, interpolated between telemetry frames. */
export function liveWaveSample(h: CrewHistory | undefined, lastFrameAt: number, speed: number): number {
  const w = h?.wave;
  if (!w || w.length < 10) return 0;
  const elapsed = (performance.now() - lastFrameAt) / (1000 / Math.max(speed, 0.25));
  const pos = w.length - 10 + Math.min(elapsed, 0.999) * 10;
  const i = Math.floor(pos), f = pos - i;
  const a = w[Math.min(i, w.length - 1)] ?? 0, b = w[Math.min(i + 1, w.length - 1)] ?? a;
  return Math.max(-2, Math.min(2, a + (b - a) * f));
}
