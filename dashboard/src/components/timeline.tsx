"use client";

import { toast } from "sonner";

import { post, useTelemetry } from "@/lib/telemetry";
import { met } from "@/lib/types";
import { cn } from "@/lib/utils";

const SEV_VAR: Record<string, string> = {
  ADVISORY: "var(--advisory)", CAUTION: "var(--caution)", WARNING: "var(--warning)", EMERGENCY: "var(--emergency)",
};

/** Tonight at a glance: one lane per crew member, scheduled events as diamonds, alerts as ticks. Click to jump. */
export function Timeline() {
  const { state, t, crew, alerts, focus, setSelected } = useTelemetry();
  const duration = state?.duration ?? 600;
  const events = state?.timeline ?? [];
  const W = 1000, LANE = 22, PAD = 56, top = 6;
  const H = top + Math.max(crew.length, 1) * LANE + 22;
  const x = (s: number) => PAD + (Math.min(s, duration) / duration) * (W - PAD - 8);
  const ticks: number[] = [];
  for (let s = 0; s <= duration; s += 60) ticks.push(s);
  const seek = (s: number) => post("/api/scenario/seek", { t: s }).catch((e) => toast.error(`Couldn't jump: ${e.message}`));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full cursor-pointer select-none"
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const px = ((e.clientX - r.left) / r.width) * W;
        if (px >= PAD) seek(Math.round(((px - PAD) / (W - PAD - 8)) * duration));
      }}
      role="slider" aria-label="Scenario time. Click to jump." tabIndex={0}
      aria-valuemin={0} aria-valuemax={duration} aria-valuenow={t} aria-valuetext={`T+${met(t)}`}
      onKeyDown={(e) => {
        if (e.key === "ArrowRight") seek(Math.min(duration - 1, t + 30));
        if (e.key === "ArrowLeft") seek(Math.max(0, t - 30));
      }}>
      <rect x={PAD} y={top - 2} width={Math.max(0, x(t) - PAD)} height={crew.length * LANE + 4} fill="var(--brand-soft)" opacity={0.6} />
      {crew.map((c, k) => {
        const y = top + k * LANE + LANE / 2;
        return (
          <g key={c.id}>
            <text x={PAD - 10} y={y + 4} textAnchor="end" onClick={(e) => { e.stopPropagation(); setSelected(c.id); }}
              className={cn("cursor-pointer text-[10.5px] font-bold", focus === c.id ? "fill-primary" : "fill-muted-foreground")}>
              {c.id}<title>{c.name}</title>
            </text>
            <line x1={PAD} x2={W - 8} y1={y} y2={y} stroke="var(--border)" />
            {events.filter((e) => e.crew === c.id).map((e, i) => (
              <g key={i} transform={`translate(${x(e.at)} ${y})`}>
                <rect x={-4} y={-4} width={8} height={8} transform="rotate(45)" strokeWidth={1.4}
                  fill={e.injected ? "var(--muted-foreground)" : "var(--card)"} stroke="var(--muted-foreground)" />
                <title>{`T+${met(e.at)} ${e.label}`}</title>
              </g>
            ))}
            {alerts.filter((a) => a.crew_id === c.id).map((a) => (
              <g key={a.id}>
                <rect x={x(a.t) - 1.5} y={y - 8} width={3} height={16} rx={1.5} fill={SEV_VAR[a.severity]} />
                <title>{`T+${met(a.t)} ${a.severity.toLowerCase()}: ${a.title}`}</title>
              </g>
            ))}
          </g>
        );
      })}
      {ticks.map((s) => (
        <text key={s} x={x(s)} y={H - 4} textAnchor="middle" className="fill-muted-foreground text-[9.5px]">{s / 60}:00</text>
      ))}
      <line x1={x(t)} x2={x(t)} y1={0} y2={H - 16} stroke="var(--brand)" strokeWidth={2.5} />
    </svg>
  );
}
