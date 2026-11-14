"use client";

import { useEffect, useRef } from "react";

import { doing, placeName, reading, ROLE, sevClass, STATUS_WORD } from "@/lib/crew";
import { useTelemetry } from "@/lib/telemetry";
import { cn } from "@/lib/utils";

const WINDOW = 100; // samples on screen: 10 s at 10 Hz
const STATUS_VAR: Record<string, string> = {
  NOMINAL: "--brand", ADVISORY: "--advisory", CAUTION: "--caution", WARNING: "--warning", EMERGENCY: "--emergency",
};

/** Movement energy per 10 Hz row, from frame-to-frame change across the 16 subcarriers. */
function movementTrace(csi: Uint8Array[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < csi.length; i++) {
    let s = 0;
    for (let c = 0; c < csi[i].length; c++) s += Math.abs(csi[i][c] - csi[i - 1][c]);
    out.push(Math.min(1, Math.log1p(s / csi[i].length) / 4.2));
  }
  return out;
}

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * Every crew member's live signal. One animation loop draws every tile so the
 * waveforms scroll together, interpolated between the 1 Hz telemetry frames.
 */
export function BreathStrip({ className }: { className?: string }) {
  const { crew, frames, framesRef, history, lastFrameAt, speedRef, modules, focus, setSelected, running } = useTelemetry();
  const canvases = useRef<Record<string, HTMLCanvasElement | null>>({});
  const runningRef = useRef(running);
  useEffect(() => { runningRef.current = running; }, [running]);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let palette: Record<string, string> = {};
    const readPalette = () => {
      const cs = getComputedStyle(document.documentElement);
      palette = Object.fromEntries(["--border", ...Object.values(STATUS_VAR)].map((v) => [v, cs.getPropertyValue(v).trim() || "#0264cc"]));
    };
    readPalette();
    const themeWatch = new MutationObserver(readPalette);
    themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "style"] });
    let visible = true;
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; });
    const list = Object.values(canvases.current)[0]?.closest("ul");
    if (list) io.observe(list);
    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!visible) return;
      const col = (v: string) => palette[v] ?? "#0264cc";
      const rule = col("--border");
      const elapsed = (performance.now() - lastFrameAt.current) / (1000 / Math.max(speedRef.current, 0.25));
      const frac = reduce || !runningRef.current ? 1 : Math.min(Math.max(elapsed, 0), 1);
      for (const [id, c] of Object.entries(canvases.current)) {
        if (!c) continue;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = c.clientWidth, h = c.clientHeight;
        if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
          c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
        }
        const ctx = c.getContext("2d")!;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        const f = framesRef.current[id];
        const hist = history.current[id];
        const breath = !!f && f.br_available !== false && (hist?.wave.length ?? 0) > 0;
        const samples = breath ? hist!.wave : movementTrace(hist?.csi ?? []);
        const color = col(STATUS_VAR[f?.status ?? "NOMINAL"]);
        const mid = h / 2;
        ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.setLineDash([2, 5]);
        ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke(); ctx.setLineDash([]);
        if (samples.length < 4) continue;

        const now = Math.max(samples.length - 10 + frac * 10, 1);
        const start = now - WINDOW, dx = w / WINDOW;
        const y = (v: number) => (breath ? mid - Math.max(-2.2, Math.min(2.2, v)) * (h / 5.2) : h - 4 - Math.min(1, v) * (h - 8));
        const at = (p: number) => {
          const i = Math.floor(p), t = p - i;
          const a = samples[Math.max(0, Math.min(samples.length - 1, i))], b = samples[Math.max(0, Math.min(samples.length - 1, i + 1))];
          return a + (b - a) * t;
        };
        const first = Math.max(Math.ceil(start), 0);
        const reliable = !f || f.br_conf >= 0.45 || !breath;
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        if (breath) {
          ctx.strokeStyle = color; ctx.globalAlpha = reliable ? 1 : 0.4; ctx.lineWidth = 2.25;
          ctx.beginPath();
          for (let p = first; p <= now; p += 1) {
            const x = (p - start) * dx;
            if (p === first) ctx.moveTo(x, y(at(p))); else ctx.lineTo(x, y(at(p)));
          }
          ctx.lineTo(w, y(at(now)));
          ctx.stroke();
        } else {
          ctx.fillStyle = color; ctx.globalAlpha = 0.5;
          for (let p = first; p <= now; p += 1) {
            const x = (p - start) * dx, top = y(at(p));
            ctx.fillRect(x - dx * 0.2, top, Math.max(1.5, dx * 0.4), h - 4 - top);
          }
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(w - 4, y(at(now)), 3.5, 0, Math.PI * 2); ctx.fill();
      }
    };
    raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); themeWatch.disconnect(); io.disconnect(); };
  }, [framesRef, history, lastFrameAt, speedRef, crew.length]);

  if (!crew.length) return <p className="px-2 py-6 text-sm text-muted-foreground">Waiting for the station to connect…</p>;

  return (
    <ul className={cn("grid gap-2", className)} aria-label="Crew, live">
      {crew.map((c, k) => {
        const f = frames[c.id];
        const r = reading(f, history.current[c.id]?.wave.length ?? 0);
        const st = f?.status ?? "NOMINAL";
        const on = focus === c.id;
        const alarm = st !== "NOMINAL";
        return (
          <li key={c.id}>
            <button type="button" onClick={() => setSelected(c.id)} aria-pressed={on}
              aria-label={`${c.name}, ${ROLE[c.id] ?? "crew"}. ${r.line}. ${STATUS_WORD[st]}. Key ${k + 1}.`}
              className={cn(sevClass(st),
                "grid h-full w-full grid-cols-[minmax(0,1fr)_auto] content-between gap-x-3 gap-y-2 rounded-[18px] bg-sunk px-3.5 pb-2.5 pt-3.5 text-start outline-none transition-colors hover:bg-brand-soft/60 focus-visible:ring-3 focus-visible:ring-ring/50",
                on && "bg-card ring-2 ring-primary",
                alarm && "bg-[color-mix(in_oklch,var(--sev)_10%,var(--card))]",
                alarm && on && "ring-(--sev)")}>
              <span className="flex min-w-0 items-center gap-2.5">
                <span className={cn("grid size-10 shrink-0 place-items-center rounded-full text-[11px] font-extrabold",
                  alarm ? "bg-(--sev) text-white" : "bg-primary text-primary-foreground")}>{c.id}</span>
                <span className="grid min-w-0">
                  <span className="text-[14.5px] font-bold leading-tight">{c.name}</span>
                  <span className="text-xs leading-snug text-muted-foreground">{f ? `${capital(doing(f))}, ${placeName(modules, f.module)}` : ROLE[c.id] ?? "Crew"}</span>
                </span>
              </span>
              <span className="grid content-center justify-items-end text-end">
                <span className={cn("text-[22px] font-extrabold leading-tight tracking-tight", r.kind === "activity" && "text-sm font-bold", !r.reliable && "text-muted-foreground")}>
                  {r.kind === "activity" ? capital(r.value) : r.value}
                  {r.unit && <small className="ms-1 text-[11px] font-semibold tracking-normal text-muted-foreground">{r.unit}</small>}
                </span>
                {alarm && <span className="max-w-[16ch] text-xs font-bold text-(--sev-ink)">{STATUS_WORD[st]}</span>}
              </span>
              <canvas ref={(el) => { canvases.current[c.id] = el; }} aria-hidden="true" className="col-span-2 block h-11 w-full" />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
