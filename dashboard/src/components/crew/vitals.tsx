"use client";

import { useEffect, useRef } from "react";
import { Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { CONF_MIN } from "@/lib/crew";
import { CSI_ROWS, CSI_SUB, liveWaveSample, type CrewHistory } from "@/lib/telemetry";

/* ---------------------------------------------------------------- raw Wi-Fi channel, as a heatmap */
function hexToRgb(h: string): [number, number, number] {
  const m = h.replace("#", "");
  const n = parseInt(m.length === 3 ? m.split("").map((c) => c + c).join("") : m, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function CsiHeatmap({ h, tick, className }: { h?: CrewHistory; tick: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const cs = getComputedStyle(document.documentElement);
    const stops = ["--heat-0", "--heat-1", "--heat-2"].map((v) => hexToRgb(cs.getPropertyValue(v).trim() || "#888888"));
    const img = ctx.createImageData(CSI_ROWS, CSI_SUB);
    const rows = h?.csi ?? [];
    const start = CSI_ROWS - rows.length;
    for (let x = 0; x < CSI_ROWS; x++) {
      const row = rows[x - start];
      for (let y = 0; y < CSI_SUB; y++) {
        const v = row ? row[y] / 255 : 0.5;
        const [a, b, f] = v < 0.5 ? [stops[0], stops[1], v / 0.5] : [stops[1], stops[2], (v - 0.5) / 0.5];
        const i = (y * CSI_ROWS + x) * 4;
        img.data[i] = a[0] + (b[0] - a[0]) * f;
        img.data[i + 1] = a[1] + (b[1] - a[1]) * f;
        img.data[i + 2] = a[2] + (b[2] - a[2]) * f;
        img.data[i + 3] = row ? 255 : 50;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [h, tick]);
  return (
    <canvas ref={ref} width={CSI_ROWS} height={CSI_SUB} className={className ?? "pixelated block h-24 w-full rounded-2xl bg-sunk"}
      aria-label="Wi-Fi channel amplitude per subcarrier over the last 12 seconds" />
  );
}

/* ---------------------------------------------------------------- breathing rate, last 90 s */
export function RateChart({ h, limits, tick }: { h?: CrewHistory; limits: [number, number]; tick: number }) {
  void tick; // re-render each telemetry frame
  const vals = (h?.bpm ?? []).slice(-90), confs = (h?.conf ?? []).slice(-90), gts = (h?.gt ?? []).slice(-90);
  const pad = 90 - vals.length;
  const data = Array.from({ length: 90 }, (_, i) => {
    const k = i - pad;
    const v = k >= 0 ? vals[k] : null;
    return {
      s: i - 89,
      wifi: k >= 0 && v != null && confs[k] >= CONF_MIN ? Number(v.toFixed(1)) : null,
      belt: k >= 0 && gts[k] != null ? Number(gts[k]!.toFixed(1)) : null,
    };
  });
  const hasBelt = data.some((d) => d.belt != null);
  return (
    <figure>
      <div className="h-[118px]" role="img" aria-label={`Breathing rate over the last 90 seconds; expected ${limits[0]} to ${limits[1]} breaths a minute`}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: 6 }}>
            <XAxis dataKey="s" hide />
            <YAxis hide domain={[Math.max(0, limits[0] - 6), limits[1] + 8]} />
            <ReferenceArea y1={limits[0]} y2={limits[1]} fill="var(--brand-soft)" fillOpacity={1} radius={10} />
            <Tooltip cursor={{ stroke: "var(--border)" }} isAnimationActive={false}
              contentStyle={{ borderRadius: 12, border: "none", background: "var(--ink)", color: "var(--background)", fontSize: 12 }}
              labelFormatter={(s) => `${-Number(s)} s ago`} formatter={(v, n) => [`${v} br/min`, n === "wifi" ? "Wi-Fi" : "Chest belt"]} />
            {hasBelt && <Line dataKey="belt" stroke="var(--muted-foreground)" strokeDasharray="4 4" strokeWidth={1.5} dot={false} isAnimationActive={false} />}
            <Line dataKey="wifi" stroke="var(--brand)" strokeWidth={2.5} dot={false} isAnimationActive={false} type="monotone" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5"><i className="h-[3px] w-3.5 rounded bg-brand" />Wi-Fi</span>
        {hasBelt && <span className="inline-flex items-center gap-1.5"><i className="w-3.5 border-t-[1.5px] border-dashed border-muted-foreground" />Chest belt</span>}
        <span className="inline-flex items-center gap-1.5"><i className="h-2.5 w-3.5 rounded-sm bg-brand-soft" />Expected {limits[0]}–{limits[1]}</span>
      </figcaption>
    </figure>
  );
}

/* ---------------------------------------------------------------- lungs that move with the live waveform */
export function Lungs({ id, history, lastFrameAt, speedRef }: {
  id: string | null; history: React.RefObject<Record<string, CrewHistory>>;
  lastFrameAt: React.RefObject<number>; speedRef: React.RefObject<number>;
}) {
  const ref = useRef<SVGGElement>(null);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    const loop = () => {
      const w = id ? liveWaveSample(history.current[id], lastFrameAt.current, speedRef.current) : 0;
      if (ref.current) ref.current.style.transform = `scale(${1 + 0.06 * w}, ${1 + 0.03 * w})`;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [id, history, lastFrameAt, speedRef]);
  return (
    <svg viewBox="0 0 160 150" className="h-[120px] w-[128px] overflow-visible" aria-hidden="true">
      <path d="M80 8v44M80 52l-16 14M80 52l16 14" fill="none" stroke="var(--lung)" strokeWidth={7} strokeLinecap="round" />
      <g ref={ref} style={{ transformOrigin: "80px 70px", transformBox: "view-box" }}>
        <path fill="var(--lung)" d="M66 60c-10-6-26 0-34 16-10 20-12 46-4 58 6 9 22 8 32 2 10-6 12-18 12-34V70c0-5-2-8-6-10z" />
        <path fill="color-mix(in oklch, var(--lung) 85%, var(--lung-hi))" d="M94 60c10-6 26 0 34 16 10 20 12 46 4 58-6 9-22 8-32 2-10-6-12-18-12-34V70c0-5 2-8 6-10z" />
      </g>
    </svg>
  );
}
