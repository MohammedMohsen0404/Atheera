"use client";

import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import { useCallback, useRef, useState } from "react";

import { ArrowsClockwise, Crosshair, PersonSimpleWalk } from "@/components/icons";
import { CsiHeatmap, RateChart } from "@/components/crew/vitals";
import { SceneBoundary } from "@/components/scene-boundary";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { doing, LIMITS, placeName, reading, sevClass, STATUS_WORD } from "@/lib/crew";
import { CSI_ROWS, useTelemetry } from "@/lib/telemetry";
import { prettyActivity } from "@/lib/types";
import { cn } from "@/lib/utils";

const Station3D = dynamic(() => import("@/scene/station-3d").then((m) => m.Station3D), {
  ssr: false,
  loading: () => <div className="absolute inset-0 grid place-items-center text-sm text-slate-400">Loading the station…</div>,
});

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function WaveTrace({ samples }: { samples: number[] }) {
  if (samples.length < 2) return <p className="text-sm text-muted-foreground">No breathing signal in this room.</p>;
  const w = 300, hgt = 54;
  const d = samples
    .map((v, i) => `${i ? "L" : "M"}${((i / (CSI_ROWS - 1)) * w + (CSI_ROWS - samples.length) * (w / (CSI_ROWS - 1))).toFixed(1)} ${(hgt / 2 - Math.max(-2, Math.min(2, v)) * (hgt / 4.6)).toFixed(1)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${hgt}`} className="block h-[54px] w-full" preserveAspectRatio="none" aria-hidden="true">
      <line x1="0" x2={w} y1={hgt / 2} y2={hgt / 2} stroke="var(--border)" strokeDasharray="2 4" />
      <path d={d} fill="none" stroke="var(--brand)" strokeWidth={2.2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-2 rounded-[18px] bg-sunk p-3.5">
      <h3 className="text-sm font-bold text-muted-foreground">{title}</h3>
      {children}
    </section>
  );
}

export default function StationPage() {
  const tel = useTelemetry();
  const { crew, frames, framesRef, history, lastFrameAt, speedRef, running, selected, setSelected, focus, modules, t } = tel;
  const { resolvedTheme } = useTheme();
  const labelEls = useRef<Record<string, HTMLDivElement | null>>({});
  const [orbit, setOrbit] = useState(false);
  const [camKey, setCamKey] = useState(0);
  const [walk, setWalk] = useState(false);
  // stable, so the walk-through isn't restarted by every telemetry update
  const exitWalk = useCallback(() => { setWalk(false); setCamKey((k) => k + 1); }, []);
  const f = focus ? frames[focus] : undefined;
  const h = focus ? history.current[focus] : undefined;
  const r = reading(f, h?.wave.length ?? 0);
  const probs = f?.activity_probs ? Object.entries(f.activity_probs).sort((a, b) => b[1] - a[1]).slice(0, 4) : [];

  return (
    <div className="grid gap-5 lg:h-[calc(100vh-130px)] lg:min-h-[560px] lg:grid-cols-[minmax(0,1fr)_360px] lg:grid-rows-[minmax(0,1fr)]">
      <section className="relative h-[62vh] overflow-hidden rounded-[22px] bg-space shadow-soft lg:h-full" data-tour="station-view" aria-label="Station view">
        <div className="absolute inset-0">
          <SceneBoundary>
          <Station3D crew={crew} frames={frames} framesRef={framesRef} history={history} lastFrameAt={lastFrameAt}
            speedRef={speedRef} running={running} selected={selected} onSelect={setSelected} theme={resolvedTheme ?? "light"}
            labelEls={labelEls} autoRotate={orbit} resetKey={camKey} walk={walk} onExitWalk={exitWalk} />
          </SceneBoundary>
        </div>

        <div className="pointer-events-none absolute inset-0">
          {crew.map((c) => {
            const fr = frames[c.id];
            if (!fr) return null;
            const rr = reading(fr, history.current[c.id]?.wave.length ?? 0);
            return (
              <button key={c.id} ref={(el) => { labelEls.current[c.id] = el as unknown as HTMLDivElement; }} onClick={() => setSelected(c.id)}
                className={cn(sevClass(fr.status), "pointer-events-auto absolute left-0 top-0 inline-flex items-center gap-2 whitespace-nowrap rounded-full bg-card px-3 py-1.5 text-[13px] shadow-float",
                  focus === c.id && "z-10 ring-2 ring-primary", fr.status !== "NOMINAL" && "z-10 ring-2 ring-(--sev)")}>
                <i className="size-2 rounded-full bg-(--sev)" /><b title={c.name}>{focus === c.id ? c.name : c.id}</b>
                <span className="text-muted-foreground">{rr.kind === "breathing" ? (rr.reliable ? `${rr.value} br/min` : "unclear") : rr.value}</span>
              </button>
            );
          })}
        </div>

        <div data-tour="station-tools" className="absolute start-4 top-4 flex gap-2">
          <Button size="sm" variant="secondary" className="bg-white/90 text-slate-900 shadow-float hover:bg-white" onClick={() => { setSelected(null); setWalk(false); setCamKey((k) => k + 1); }}>
            <Crosshair weight="bold" />Whole station
          </Button>
          <Button size="sm" variant="secondary" aria-pressed={orbit} disabled={walk} onClick={() => setOrbit((o) => !o)}
            className={cn("shadow-float", orbit ? "bg-primary text-primary-foreground hover:bg-primary" : "bg-white/90 text-slate-900 hover:bg-white")}>
            <ArrowsClockwise weight="bold" />{orbit ? "Stop orbit" : "Auto orbit"}
          </Button>
          <Button size="sm" variant="secondary" aria-pressed={walk} onClick={() => { if (walk) exitWalk(); else { setOrbit(false); setWalk(true); } }}
            className={cn("shadow-float", walk ? "bg-primary text-primary-foreground hover:bg-primary" : "bg-white/90 text-slate-900 hover:bg-white")}>
            <PersonSimpleWalk weight="bold" />{walk ? "Leave walk-through" : "Walk through"}
          </Button>
        </div>

        {walk && (
          <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-4">
            <p className="max-w-xl rounded-2xl bg-slate-950/75 px-4 py-2.5 text-center text-sm text-slate-100 backdrop-blur">
              Move with <b>W A S D</b> or the arrow keys. <b>R</b> and <b>F</b> go up and down. Drag to look around, hold <b>Shift</b> to go faster, and press <b>Esc</b> to leave.
            </p>
          </div>
        )}

        <ul aria-label="Legend" className={cn("absolute bottom-4 end-4 hidden max-w-[220px] gap-1.5 rounded-2xl bg-slate-950/70 p-3 text-xs text-slate-200 backdrop-blur sm:grid", walk && "sm:hidden")}>
          <li className="flex items-center gap-2"><i className="size-2.5 shrink-0 rounded-full bg-brand-2 ring-3 ring-brand-2/30" />Wi-Fi signal passing through a crew member</li>
          <li className="flex items-center gap-2"><i className="size-2.5 shrink-0 rounded-full bg-nominal" />Normal</li>
          <li className="flex items-center gap-2"><i className="size-2.5 shrink-0 rounded-full bg-caution" />Caution</li>
          <li className="flex items-center gap-2"><i className="size-2.5 shrink-0 rounded-full bg-emergency" />Warning or emergency: the module lights flash</li>
        </ul>
        {!walk && <p className="absolute bottom-4 start-5 text-xs text-slate-400">Drag to orbit, Ctrl + scroll to zoom. Modules are schematic, not to scale.</p>}
      </section>

      <Card data-tour="station-panel" className="gap-4 overflow-y-auto px-5">
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Crew member">
          {crew.map((c) => (
            <Button key={c.id} size="sm" aria-pressed={focus === c.id} onClick={() => setSelected(c.id)}
              className={cn(sevClass(frames[c.id]?.status), "gap-2", focus === c.id ? "" : "bg-sunk text-foreground hover:bg-brand-soft")}>
              <i className="size-2 rounded-full bg-(--sev)" />{c.name}
            </Button>
          ))}
        </div>
        {f ? (
          <>
            <header className="grid gap-1">
              <h2 className="text-[22px] font-extrabold">{f.name}</h2>
              <p className="text-sm text-muted-foreground">{capital(doing(f))} in {placeName(modules, f.module).toLowerCase()}</p>
              <p className={cn(sevClass(f.status), "mt-1.5 flex items-center gap-2 rounded-2xl bg-sunk px-3 py-2.5 text-sm font-semibold")}>
                <i className="size-2 shrink-0 rounded-full bg-(--sev)" />{STATUS_WORD[f.status]}. {r.line}.
              </p>
            </header>
            <Section title="Wi-Fi channel"><CsiHeatmap h={h} tick={t} className="pixelated block h-20 w-full rounded-xl bg-card" /></Section>
            {r.kind === "breathing" ? (
              <>
                <Section title="Breathing signal"><WaveTrace samples={h?.wave ?? []} /></Section>
                <Section title="Rate">
                  <p className="text-sm text-muted-foreground">
                    <span className={cn("text-3xl font-extrabold tracking-tight text-foreground", !r.reliable && "text-muted-foreground")}>{f.bpm != null ? f.bpm.toFixed(1) : "—"}</span>
                    {" "}br/min, {Math.round(f.br_conf * 100)}% clear{f.gt_bpm != null ? `, chest belt ${f.gt_bpm.toFixed(1)}` : ""}
                  </p>
                  <RateChart h={h} limits={LIMITS[f.context] ?? LIMITS.REST} tick={t} />
                </Section>
              </>
            ) : (
              <Section title="Activity">
                <ul className="grid gap-2">
                  {probs.map(([k, v]) => (
                    <li key={k} className="grid grid-cols-[84px_1fr_38px] items-center gap-2.5 text-sm">
                      <span>{capital(prettyActivity(k))}</span>
                      <span className="h-2.5 overflow-hidden rounded-full bg-card">
                        <span className={cn("block h-full rounded-full", k === "fall" ? "bg-emergency" : "bg-brand-2")} style={{ width: `${Math.round(v * 100)}%` }} />
                      </span>
                      <span className="text-end text-muted-foreground">{Math.round(v * 100)}%</span>
                    </li>
                  ))}
                </ul>
              </Section>
            )}
            <p className="break-all text-xs text-muted-foreground" title={f.source}>{f.synthetic ? "Synthetic placeholder track" : "Recorded data"}: {f.track}</p>
          </>
        ) : <p className="py-3 text-sm text-muted-foreground">Press play to start the replay, then pick a crew member.</p>}
      </Card>
    </div>
  );
}
