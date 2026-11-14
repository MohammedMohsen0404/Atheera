"use client";

import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { BreathStrip } from "@/components/crew/breath-strip";
import { RestLine } from "@/components/crew/rest";
import { CsiHeatmap, Lungs } from "@/components/crew/vitals";
import { SceneBoundary } from "@/components/scene-boundary";
import { Timeline } from "@/components/timeline";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { doing, LIMITS, placeName, reading, sevClass, STATUS_WORD } from "@/lib/crew";
import { post, useTelemetry } from "@/lib/telemetry";
import { met, prettyActivity } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { AnchorName } from "@/scene/astronaut-hero";

const AstronautHero = dynamic(() => import("@/scene/astronaut-hero").then((m) => m.AstronautHero), {
  ssr: false,
  loading: () => <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">Loading the figure…</div>,
});

const SCENARIOS: { kind: string; label: string; what: string }[] = [
  { kind: "fall", label: "Simulate a fall", what: "Should raise an emergency within seconds" },
  { kind: "breath_slow", label: "Slow breathing", what: "Should raise a caution once breathing drops below the sleep range" },
  { kind: "interference", label: "Radio interference", what: "A noisier signal with a person moving nearby; breathing is still read, so at most an advisory, never an alarm" },
  { kind: "exercise", label: "Start exercise", what: "Faster breathing is expected, so no alarm" },
  { kind: "recover", label: "Back to schedule", what: "Clears the simulated event" },
];

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function Callout({ side, setRef, children }: { side: "left" | "right" | "below"; setRef: (el: HTMLDivElement | null) => void; children: React.ReactNode }) {
  return (
    <div ref={setRef} className={cn("absolute flex items-center gap-2 whitespace-nowrap",
      side === "right" && "-translate-x-[7px] -translate-y-1/2",
      side === "left" && "flex-row-reverse -translate-y-1/2 translate-x-[calc(-100%+7px)]",
      side === "below" && "-translate-x-1/2 -translate-y-[7px] flex-col")}>
      <i className="relative size-3.5 shrink-0 rounded-full bg-card ring-3 ring-primary/30 after:absolute after:inset-1 after:rounded-full after:bg-primary" />
      <span className={cn("rounded-full bg-card px-3 py-1.5 text-xs text-muted-foreground shadow-soft [&_b]:text-foreground",
        side === "right" && "ms-4", side === "left" && "me-4", side === "below" && "mt-2")}>{children}</span>
    </div>
  );
}

export default function CrewPage() {
  const tel = useTelemetry();
  const { state, frames, framesRef, history, lastFrameAt, speedRef, notice, ack, setSelected, t, connected, focus, crew, modules } = tel;
  const [busy, setBusy] = useState<string | null>(null);
  const { resolvedTheme } = useTheme();
  const callouts = useRef<Record<AnchorName, HTMLDivElement | null>>({ chest: null, head: null, hand: null, router: null });
  const f = focus ? frames[focus] : undefined;
  const h = focus ? history.current[focus] : undefined;
  const r = reading(f, h?.wave.length ?? 0);
  const limits = LIMITS[f?.context ?? "REST"] ?? LIMITS.REST;
  const kinds = state?.inject_kinds ?? SCENARIOS.map((s) => s.kind);
  const hasFrames = Object.keys(frames).length > 0;
  const doingNow = f ? doing(f) : "resting";

  const nameOf = (id: string) => crew.find((c) => c.id === id)?.name ?? id;

  const inject = async (kind: string) => {
    if (!focus) return;
    setBusy(kind);
    try { await post("/api/inject", { crew_id: focus, kind }); }
    catch (e) { toast.error(`The event wasn't sent: ${(e as Error).message}. Check that the backend is running.`); }
    finally { setBusy(null); }
  };

  const probs = f?.activity_probs ? Object.entries(f.activity_probs).sort((a, b) => b[1] - a[1]).slice(0, 4) : [];
  const motionWord = !f ? "—" : f.anomaly > 0.8 ? "Unusual" : f.motion > 0.03 ? "Active" : "Calm";

  // The chest belt only reports now and then, so keep showing its last reading instead of letting the line blink.
  // It adds how old the reading is once it gets stale.
  let belt: { value: number; age: number } | null = null;
  if (h) {
    for (let i = h.gt.length - 1; i >= 0; i--) {
      const v = h.gt[i];
      if (v != null) { belt = { value: v, age: (h.t[h.t.length - 1] ?? 0) - h.t[i] }; break; }
    }
  }

  return (
    <div className="grid gap-5">
      <header className="relative flex flex-wrap items-end justify-between gap-x-6 gap-y-3 px-1">
        <div>
          <h1 className="text-[28px] font-extrabold tracking-tight sm:text-[30px]">Good evening, flight surgeon</h1>
          <p className="mt-1 text-muted-foreground">{state?.scenario ?? "Night shift"}, {crew.length || 4} crew aboard. Everything here is read from the station&apos;s Wi-Fi.</p>
        </div>
        {/* The one alert notice: a few seconds beside the heading. On wide screens it floats, so nothing moves. */}
        {notice && (
          <div role="alert" className={cn(sevClass(notice.severity),
            "animate-in fade-in-0 duration-300 relative flex flex-wrap items-center gap-x-4 gap-y-2 rounded-3xl bg-(--sev) py-2 pe-2 ps-5 text-white shadow-float sm:rounded-full dark:bg-[color-mix(in_oklch,var(--sev)_62%,black)] xl:absolute xl:bottom-0 xl:end-1",
            notice.severity === "EMERGENCY" && "alarm-pulse")}>
            <p className="text-sm">
              <b className="me-1.5">{notice.severity === "EMERGENCY" ? "Emergency" : "Warning"}</b>
              {nameOf(notice.crew_id)}: {notice.title.toLowerCase()}
            </p>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" className="border-white/60 bg-transparent text-white hover:bg-white/15 hover:text-white"
                onClick={() => setSelected(notice.crew_id)}>See {nameOf(notice.crew_id)}</Button>
              <Button size="sm" className="bg-white text-slate-900 hover:bg-white/90" onClick={() => ack(notice.id)}>Acknowledge</Button>
            </div>
          </div>
        )}
      </header>

      {!hasFrames && connected && (
        <Card className="flex-row flex-wrap items-center justify-between gap-4 px-6">
          <div>
            <h2 className="text-lg font-bold">Watch a night shift on the station</h2>
            <p className="text-sm text-muted-foreground">This is a demo. Atheera reads real recorded Wi-Fi data from seven crew members, ten minutes played at {state?.speed ?? 4}× speed, and shows breathing, activity and alerts as if it were live.</p>
          </div>
          <Button size="lg" onClick={() => post("/api/scenario/start")}>Play the demo</Button>
        </Card>
      )}

      <div className="grid items-stretch gap-5 xl:grid-cols-[minmax(300px,0.95fr)_minmax(0,1.5fr)_minmax(280px,0.95fr)] lg:grid-cols-[minmax(280px,1fr)_minmax(0,1.6fr)]">
        <div className="grid min-w-0 gap-5">
          <Card data-tour="wifi" className="px-5">
            <CardHeader className="flex items-center justify-between px-0">
              <CardTitle className="text-base font-bold">Wi-Fi signal</CardTitle>
              {f && r.kind === "breathing" && (
                <Badge variant="secondary" className={cn(!r.reliable && "bg-[color-mix(in_oklch,var(--caution)_18%,var(--card))] text-caution-ink")}>
                  {Math.round(f.br_conf * 100)}% clear
                </Badge>
              )}
            </CardHeader>
            <CsiHeatmap h={h} tick={t} />
            <CardDescription>How the signal ripples around {f?.name ?? "the crew member"}, last 12 seconds</CardDescription>
          </Card>
          <Card data-tour="activity" className="px-5">
            <CardHeader className="px-0"><CardTitle className="text-base font-bold">Activity</CardTitle></CardHeader>
            {probs.length > 0 ? (
              <ul className="grid gap-2.5">
                {probs.map(([k, v]) => (
                  <li key={k} className="grid grid-cols-[84px_1fr_38px] items-center gap-2.5 text-sm">
                    <span>{capital(prettyActivity(k))}</span>
                    <span className="h-2.5 overflow-hidden rounded-full bg-sunk">
                      <span className={cn("block h-full rounded-full", k === "fall" ? "bg-emergency" : "bg-brand-2")} style={{ width: `${Math.round(v * 100)}%` }} />
                    </span>
                    <span className="text-end text-muted-foreground">{Math.round(v * 100)}%</span>
                  </li>
                ))}
              </ul>
            ) : f ? (
              <div className="grid gap-1">
                <p className="text-2xl font-extrabold tracking-tight">{capital(doingNow)}</p>
                {f.rest && f.rest.sleep_period_s > 0 ? (
                  <RestLine r={f.rest} />
                ) : (
                  <CardDescription className="mt-1">This room measures breathing, not activities, so this comes from tonight&apos;s schedule.</CardDescription>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Press play to see what the crew are doing.</p>
            )}
          </Card>
        </div>

        <div className="grid min-w-0 gap-5">
          <Card className="relative h-[470px] p-0 lg:h-auto lg:min-h-[470px] [background:radial-gradient(circle_at_50%_45%,var(--card),var(--sunk))]" data-tour="figure" aria-label={f ? `${f.name}, figure` : "Crew figure"}>
            {f && (
              <div className="absolute start-5.5 top-4.5 z-10 grid gap-0.5">
                <b className="text-xl font-extrabold tracking-tight">{f.name}</b>
                <span className="text-sm text-muted-foreground">{capital(doingNow)} in {placeName(modules, f.module).toLowerCase()}</span>
              </div>
            )}
            {f && (
              <Badge className={cn(sevClass(f.status), "absolute end-4.5 top-4.5 z-10 h-7 gap-2 px-3 text-[13px] font-bold bg-[color-mix(in_oklch,var(--sev)_13%,var(--card))] text-(--sev-ink)")}>
                <i className="size-2 rounded-full bg-(--sev)" />{STATUS_WORD[f.status]}
              </Badge>
            )}
            <div aria-hidden="true" className="pointer-events-none absolute bottom-7 left-1/2 h-7 w-[44%] -translate-x-1/2 rounded-[50%] [background:radial-gradient(closest-side,color-mix(in_oklch,var(--ink)_14%,transparent),transparent)]" />
            <SceneBoundary>
              <AstronautHero crewId={focus} framesRef={framesRef} history={history} lastFrameAt={lastFrameAt}
                speedRef={speedRef} calloutEls={callouts} theme={resolvedTheme ?? "light"} />
            </SceneBoundary>
            {f && (
              <div className="pointer-events-none absolute inset-0 z-10" aria-hidden="true">
                <Callout side="right" setRef={(el) => { callouts.current.chest = el; }}>
                  <b>{r.kind === "breathing" ? (r.reliable ? `${r.value} br/min` : "Unclear") : "No sensor"}</b> breathing
                </Callout>
                <Callout side="right" setRef={(el) => { callouts.current.head = el; }}><b>{capital(doingNow)}</b></Callout>
                <Callout side="left" setRef={(el) => { callouts.current.hand = el; }}><b>{motionWord}</b> movement</Callout>
                <Callout side="below" setRef={(el) => { callouts.current.router = el; }}><b>Wi-Fi</b> sensor</Callout>
              </div>
            )}
            {!f && <div className="absolute inset-0 grid place-items-center text-muted-foreground">Press play to bring the crew to life.</div>}
          </Card>
        </div>

        <div className="grid min-w-0 gap-5 lg:col-span-2 lg:grid-cols-2 xl:col-span-1 xl:grid-cols-1">
          <Card data-tour="breathing" className="grid grid-cols-[1fr_128px] items-center gap-x-3 gap-y-1 px-5">
            <div className="grid gap-1">
              <h2 className="text-base font-bold">Breathing</h2>
              <p className="mt-1.5 text-[46px] font-extrabold leading-none tracking-tighter">
                {r.kind === "breathing" && r.reliable ? r.value : "—"}<small className="ms-1.5 text-[13px] font-semibold tracking-normal text-muted-foreground">br/min</small>
              </p>
              <p className="text-sm text-muted-foreground">{r.kind === "breathing" ? `Expected ${limits[0]}–${limits[1]} while ${doingNow}` : "No breathing sensor in this room"}</p>
              {belt && <p className="text-sm text-muted-foreground">Chest belt says {belt.value.toFixed(1)}{belt.age >= 10 ? ` (${Math.round(belt.age)} s ago)` : ""}</p>}
            </div>
            <Lungs id={r.kind === "breathing" ? focus : null} history={history} lastFrameAt={lastFrameAt} speedRef={speedRef} />
            <p className={cn(f ? sevClass(f.status) : "sev-nominal", "col-span-2 mt-2.5 flex items-center gap-2 rounded-2xl bg-sunk px-3.5 py-2.5 text-sm font-semibold",
              f && f.status !== "NOMINAL" && "bg-[color-mix(in_oklch,var(--sev)_12%,var(--card))] text-(--sev-ink)")}>
              <i className="size-2 rounded-full bg-(--sev)" />{r.line}
            </p>
          </Card>

          <Card data-tour="scenarios" className="px-5">
            <CardHeader className="px-0">
              <CardTitle className="text-base font-bold">Try a scenario</CardTitle>
              <CardDescription>Play out an event for {f?.name ?? "the selected crew member"} and watch Atheera respond.</CardDescription>
            </CardHeader>
            <div className="flex flex-wrap gap-2">
              {SCENARIOS.filter((s) => kinds.includes(s.kind)).map((s) => (
                <Button key={s.kind} size="sm" variant={s.kind === "fall" ? "destructive" : "secondary"} title={s.what}
                  className={cn("h-9 px-3.5 text-[13px]", s.kind !== "fall" && "bg-sunk text-foreground hover:bg-brand-soft hover:text-primary")}
                  disabled={!connected || !focus || busy !== null} onClick={() => inject(s.kind)}>
                  {busy === s.kind ? "Sending…" : s.label}
                </Button>
              ))}
            </div>
            {f && <p className="break-all text-xs text-muted-foreground" title={f.source}>{f.synthetic ? "Synthetic placeholder track" : "Recorded data"}: {f.track}{f.injected ? ", with a simulated event" : ""}</p>}
          </Card>
        </div>
      </div>

      <Card data-tour="crew-list" className="px-4 pt-5 sm:px-5">
        <CardHeader className="flex items-center justify-between px-0">
          <CardTitle className="text-lg font-bold">Crew</CardTitle>
          <span className="text-sm text-muted-foreground">Tap to inspect</span>
        </CardHeader>
        <BreathStrip className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" />
      </Card>

      <Card data-tour="timeline" className="px-5">
        <CardHeader className="flex flex-wrap items-start justify-between gap-2 px-0">
          <div>
            <CardTitle className="text-base font-bold">Tonight</CardTitle>
            <CardDescription>Diamonds are scheduled events, ticks are alerts. Click to jump to that moment.</CardDescription>
          </div>
          <Badge variant="secondary" className="h-6 px-2.5 font-bold">Now T+{met(t)}</Badge>
        </CardHeader>
        <Timeline />
      </Card>
    </div>
  );
}
