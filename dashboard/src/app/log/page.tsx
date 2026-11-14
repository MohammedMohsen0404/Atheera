"use client";

import Link from "next/link";
import { useState } from "react";

import { CheckCircle, Warning } from "@/components/icons";
import { Timeline } from "@/components/timeline";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { placeName, sevClass } from "@/lib/crew";
import { useTelemetry } from "@/lib/telemetry";
import { contextWords, met, SEVERITY_ORDER, type Alert } from "@/lib/types";
import { cn } from "@/lib/utils";

type Filter = "open" | "handled" | "all";

const LEVELS = [
  { sev: "emergency", name: "Emergency", what: "Possible fall or impact.", act: "Call the crew member on the comm loop now." },
  { sev: "warning", name: "Warning", what: "A reading is clearly outside the safe range, or no breathing or movement was seen.", act: "Check in with the crew member." },
  { sev: "caution", name: "Caution", what: "A reading has drifted outside the expected range.", act: "Keep watching. Check in if it continues." },
  { sev: "advisory", name: "Advisory", what: "The signal is poor, for example from radio interference, or something moved near a resting crew member while their breathing stayed steady.", act: "No action unless it persists." },
];
const ACTION: Record<string, string> = { EMERGENCY: LEVELS[0].act, WARNING: LEVELS[1].act, CAUTION: LEVELS[2].act, ADVISORY: LEVELS[3].act };
const SEV_NAME: Record<string, string> = { EMERGENCY: "Emergency", WARNING: "Warning", CAUTION: "Caution", ADVISORY: "Advisory", NOMINAL: "Normal" };

function explain(a: Alert, name: string): string {
  const e = a.evidence as Record<string, any>;
  const held = e.persisted_s ? ` for ${Math.round(e.persisted_s)} s` : "";
  const clarity = e.confidence != null ? ` The signal was ${Math.round(e.confidence * 100)}% clear.` : "";
  const range = e.limits ? `${e.limits[0]}–${e.limits[1]} br/min expected while ${contextWords(e.context)}` : "expected range";
  switch (a.code) {
    case "br_low": return `Breathing read ${e.bpm} br/min${held}, below the ${range}.${clarity}`;
    case "br_high": return `Breathing read ${e.bpm} br/min${held}, above the ${range}.${clarity}`;
    case "fall": return `The activity model saw a fall pattern with ${Math.round((e.fall_prob ?? 0) * 100)}% confidence.`;
    case "motion": return a.severity === "ADVISORY"
      ? `Movement near ${name}${held}, while their breathing stayed steady at ${e.bpm} br/min. It is most likely someone or something nearby; it becomes a warning if the breathing reading breaks.`
      : `${name} moved far more than usual${held}.`;
    case "br_lost": return `The breathing rhythm faded from the Wi-Fi signal${held}. Usually a change of position or radio interference.`;
    case "still": return `The room is occupied, but there was no movement and no breathing rhythm${held}.`;
    default: return a.title;
  }
}

const ago = (t: number, now: number) => {
  const s = Math.max(0, Math.round(now - t));
  return s < 60 ? `${s} s ago` : `${Math.floor(s / 60)} min ago`;
};

export default function LogPage() {
  const { alerts, crew, modules, t, ack, setSelected } = useTelemetry();
  const [filter, setFilter] = useState<Filter>("open");
  const names = Object.fromEntries(crew.map((c) => [c.id, c.name]));
  const open = alerts.filter((a) => !a.acked);
  const handled = alerts.filter((a) => a.acked);
  const ackTimes = handled.filter((a) => a.acked_t != null).map((a) => (a.acked_t as number) - a.t).sort((x, y) => x - y);
  const medianAck = ackTimes.length ? ackTimes[Math.floor(ackTimes.length / 2)] : null;
  const shown = (filter === "open" ? open : filter === "handled" ? handled : alerts)
    .slice().sort((a, b) => Number(a.acked) - Number(b.acked) || SEVERITY_ORDER.indexOf(b.severity) - SEVERITY_ORDER.indexOf(a.severity) || b.t - a.t);
  const worst = open.length ? open.reduce((w, a) => (SEVERITY_ORDER.indexOf(a.severity) > SEVERITY_ORDER.indexOf(w.severity) ? a : w)) : null;
  const headline = open.length === 0
    ? (alerts.length ? "Nothing needs your attention right now." : "A quiet shift so far. No alerts.")
    : `${open.length === 1 ? "One alert needs" : `${open.length} alerts need`} your attention.`;
  const summary = [
    handled.length ? `${handled.length} handled tonight` : null,
    medianAck != null ? `acknowledged after ${Math.round(medianAck)} s (median, station time)` : null,
  ].filter(Boolean).join(", ");

  return (
    <div className="grid gap-5">
      <header className={cn(worst ? sevClass(worst.severity) : "sev-nominal", "px-1")}>
        <h1 className={cn("text-[28px] font-extrabold tracking-tight sm:text-[30px]", worst && "text-(--sev-ink)")}>{headline}</h1>
        <p className="mt-1 text-muted-foreground">{summary ? `${summary.charAt(0).toUpperCase()}${summary.slice(1)}.` : "New alerts appear here, at the top of every page and as a notification."}</p>
      </header>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_380px]">
        <section data-tour="log-list" className="grid gap-3.5" aria-live="polite">
          <Tabs value={filter} onValueChange={(v) => setFilter(v as Filter)}>
            <TabsList data-tour="log-tabs" className="h-auto rounded-full bg-card p-1.5 shadow-soft">
              <TabsTrigger value="open" className="rounded-full px-4 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Open {open.length}</TabsTrigger>
              <TabsTrigger value="handled" className="rounded-full px-4 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Handled {handled.length}</TabsTrigger>
              <TabsTrigger value="all" className="rounded-full px-4 py-2 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Everything</TabsTrigger>
            </TabsList>
          </Tabs>

          {shown.length === 0 && (
            <Card className="px-6">
              <p className="text-lg font-bold">{filter === "handled" ? "Nothing handled yet." : "All clear."}</p>
              <p className="text-muted-foreground">
                {filter === "handled" ? "Alerts you acknowledge move here." : <>Try a scenario on the <Link href="/" className="font-semibold text-primary underline-offset-4 hover:underline">Crew page</Link>, such as a simulated fall, to see an alert arrive.</>}
              </p>
            </Card>
          )}

          {shown.map((a) => {
            const name = names[a.crew_id] ?? a.crew_id;
            return (
              <Card key={a.id} className={cn(sevClass(a.severity), "grid grid-cols-[52px_minmax(0,1fr)] gap-4 px-5 sm:grid-cols-[52px_minmax(0,1fr)_auto]")}>
                <span className={cn("grid size-[52px] place-items-center rounded-2xl", a.acked ? "bg-sunk text-muted-foreground" : "bg-[color-mix(in_oklch,var(--sev)_14%,var(--card))] text-(--sev)")}>
                  {a.acked ? <CheckCircle weight="bold" className="size-6" /> : <Warning weight="fill" className="size-6" />}
                </span>
                <div className="grid gap-1">
                  <p className={cn("inline-flex items-center gap-1.5 text-xs font-bold", a.acked ? "text-muted-foreground" : "text-(--sev-ink)")}>
                    <i className="size-1.5 rounded-full bg-(--sev)" />{SEV_NAME[a.severity]}
                  </p>
                  <h2 className={cn("text-[17px] font-bold", a.acked && "text-muted-foreground")}>{name}: {a.title.charAt(0).toLowerCase() + a.title.slice(1)}</h2>
                  <p className="text-sm text-muted-foreground">{placeName(modules, a.module)}</p>
                  <p className="mt-1.5 rounded-2xl bg-sunk px-3.5 py-2.5 text-sm text-foreground/80">{explain(a, name)}</p>
                  {a.acked ? (
                    <p className="mt-1.5 text-sm text-muted-foreground">
                      Acknowledged{a.acked_t != null ? ` at T+${met(a.acked_t)}, ${Math.round(a.acked_t - a.t)} s after it fired` : ""}.
                      {a.cleared_t != null ? " The condition has cleared." : " The condition is still present."}
                    </p>
                  ) : (
                    <div className="mt-2.5 flex flex-wrap items-center justify-between gap-3">
                      <p className="text-sm font-bold">{ACTION[a.severity] ?? "Review the reading."}</p>
                      <div className="flex gap-2">
                        <Button asChild variant="outline" size="sm"><Link href="/" onClick={() => setSelected(a.crew_id)}>See {name}</Link></Button>
                        <Button size="sm" onClick={() => ack(a.id)}>Acknowledge</Button>
                      </div>
                    </div>
                  )}
                </div>
                <div className="col-start-2 grid content-start sm:col-start-3 sm:justify-items-end sm:text-end">
                  <b className="text-sm tabular-nums">T+{met(a.t)}</b>
                  <span className="text-xs text-muted-foreground">{ago(a.t, t)}</span>
                </div>
              </Card>
            );
          })}
        </section>

        <aside className="grid gap-5">
          <Card data-tour="levels" className="px-5">
            <CardHeader className="px-0"><CardTitle className="text-base font-bold">What the levels mean</CardTitle></CardHeader>
            <dl className="grid gap-2.5">
              {LEVELS.map((l) => (
                <div key={l.sev} className={cn(`sev-${l.sev}`, "rounded-2xl bg-sunk px-3.5 py-3")}>
                  <dt className="flex items-center gap-2 font-bold"><i className="size-2 rounded-full bg-(--sev)" />{l.name}</dt>
                  <dd className="ms-4 mt-0.5 text-sm text-muted-foreground">{l.what} <span className="font-semibold text-foreground/80">{l.act}</span></dd>
                </div>
              ))}
            </dl>
          </Card>
        </aside>
      </div>

      <Card className="px-5">
        <CardHeader className="px-0">
          <CardTitle className="text-base font-bold">Tonight</CardTitle>
          <CardDescription>Ticks are alerts, diamonds are scheduled events. Click to jump to that moment.</CardDescription>
        </CardHeader>
        <Timeline />
      </Card>
    </div>
  );
}
