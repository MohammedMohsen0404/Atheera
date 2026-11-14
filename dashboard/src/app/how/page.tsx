"use client";

import { useQuery } from "@tanstack/react-query";
import { Fragment } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Scatter, ComposedChart, Tooltip, XAxis, YAxis } from "recharts";

import { Broadcast, ChartLineUp, Gauge, Waveform, Bell } from "@/components/icons";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

type Evidence = Record<string, any>;

const STEPS = [
  { icon: Broadcast, title: "Listen to the Wi-Fi", text: "The station's access point measures how its signal bends around each person, on many frequencies at once." },
  { icon: Waveform, title: "Clean the signal", text: "Spikes are removed and the breathing band, 6 to 36 breaths a minute, is kept from the frequencies that carry it best." },
  { icon: ChartLineUp, title: "Read the body", text: "A breathing rate with a clarity score, and a published, pretrained model that recognises activities and falls." },
  { icon: Gauge, title: "Decide", text: "Rules compare each reading with that person's schedule and usual level. Unclear readings never raise an alarm." },
  { icon: Bell, title: "Tell the flight surgeon", text: "Alerts that show the evidence behind them and what to do next." },
];

const label = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const tooltipStyle = { borderRadius: 12, border: "none", background: "var(--ink)", color: "var(--background)", fontSize: 12 };
const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`;

/** Exact (Clopper-Pearson) 95% interval for k successes out of n, so small samples show how uncertain they are. */
function ci95(k: number, n: number): [number, number] {
  if (n <= 0) return [0, 1];
  const logFact = [0];
  for (let i = 1; i <= n; i++) logFact.push(logFact[i - 1] + Math.log(i));
  const pmf = (i: number, p: number) => Math.exp(logFact[n] - logFact[i] - logFact[n - i] + i * Math.log(p) + (n - i) * Math.log(1 - p));
  const cdf = (x: number, p: number) => { let s = 0; for (let i = 0; i <= x; i++) s += pmf(i, p); return s; };
  const solve = (f: (p: number) => number, target: number, rising: boolean) => {
    let lo = 1e-9, hi = 1 - 1e-9;
    for (let it = 0; it < 60; it++) {
      const mid = (lo + hi) / 2;
      if ((f(mid) < target) === rising) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  };
  const lower = k === 0 ? 0 : solve((p) => 1 - cdf(k - 1, p), 0.025, true);  // P(X >= k) rises with p
  const upper = k === n ? 1 : solve((p) => cdf(k, p), 0.025, false);          // P(X <= k) falls with p
  return [lower, upper];
}

/** The models, in plain terms. Facts come from the code: atheer/modules and atheer/fusion. */
const MODELS = [
  { task: "Breathing rate", model: "Signal processing, no learned model: a band-pass filter, the 30 clearest channels fused into one spectrum, and Viterbi tracking over its peaks", learned: "No" },
  { task: "Breathing clarity", model: "Isotonic regression from the strength of the spectral peak to the chance that the reading is within 2 breaths a minute", learned: "Yes, one number in and one out" },
  { task: "Activity and falls", model: "ResNet18 neural network, an ensemble of 4 (11.2 million parameters each) with calibrated probabilities", learned: "Yes, on whole UT-HAR recordings" },
  { task: "Unusual movement", model: "Robust z-score of a motion index against that person's own last 2 minutes", learned: "No" },
  { task: "Alerts", model: "Rules: context, normal ranges, a clarity gate, and a wait before alarming", learned: "No" },
];

function TableView({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <details className="text-sm">
      <summary className="w-fit cursor-pointer font-semibold text-muted-foreground hover:text-foreground">Show as table</summary>
      <table className="mt-2.5 w-full border-collapse text-[13px]">
        <thead><tr>{head.map((h) => <th key={h} className="border-b px-2 py-1.5 text-start text-muted-foreground">{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="border-b px-2 py-1.5">{c}</td>)}</tr>)}</tbody>
      </table>
    </details>
  );
}

function HBar({ data, format, max }: { data: { key: string; value: number }[]; format: (v: number) => string; max?: number }) {
  return (
    <div style={{ height: data.length * 34 + 8 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 56, bottom: 0, left: 0 }} barCategoryGap={8}>
          <XAxis type="number" hide domain={[0, max ?? "auto"]} />
          <YAxis type="category" dataKey="key" width={104} tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 13 }} tickFormatter={label} />
          <Tooltip cursor={{ fill: "var(--sunk)" }} contentStyle={tooltipStyle} formatter={(v) => [format(Number(v)), ""]} labelFormatter={(k) => label(String(k))} />
          <Bar dataKey="value" radius={[6, 6, 6, 6]} fill="var(--brand-2)" background={{ fill: "var(--sunk)", radius: 6 }} isAnimationActive={false}>
            <LabelList dataKey="value" position="right" formatter={(v) => format(Number(v))} className="fill-foreground text-[13px] font-bold" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Confusion({ classes, matrix }: { classes: string[]; matrix: number[][] }) {
  const rows = matrix.map((r) => {
    const n = r.reduce((a, b) => a + b, 0) || 1;
    return r.map((v) => ({ v, p: v / n, n }));
  });
  return (
    <div className="grid gap-1 text-xs" style={{ gridTemplateColumns: `auto repeat(${classes.length}, minmax(0, 1fr))` }}>
      <span className="self-end text-[11px] text-muted-foreground">Actual ↓ predicted →</span>
      {classes.map((c) => <span key={c} className="self-end text-center leading-tight text-muted-foreground">{label(c)}</span>)}
      {rows.map((r, i) => (
        <Fragment key={i}>
          <span className="self-center whitespace-nowrap pe-1.5 text-foreground/80">{label(classes[i])}</span>
          {r.map((cell, j) => (
            <span key={j} title={`${label(classes[i])} → ${label(classes[j])}: ${cell.v} of ${cell.n} windows (${(cell.p * 100).toFixed(1)}%)`}
              className="grid aspect-[1.4] place-items-center rounded-lg text-xs font-bold"
              style={{
                background: cell.v ? `color-mix(in oklch, var(--brand) ${Math.round(8 + cell.p * 92)}%, var(--card))` : "var(--sunk)",
                color: cell.p > 0.55 ? "var(--primary-foreground)" : "var(--foreground)",
              }}>
              {cell.v ? (cell.p >= 0.995 ? "100" : cell.p >= 0.01 ? Math.round(cell.p * 100) : "<1") : ""}
            </span>
          ))}
        </Fragment>
      ))}
    </div>
  );
}

export default function HowPage() {
  const q = useQuery({
    queryKey: ["evidence"],
    queryFn: async () => {
      const r = await fetch("/api/evidence");
      if (!r.ok) throw new Error(`the server answered ${r.status}`);
      return (await r.json()) as Evidence;
    },
  });
  const ev = q.data;
  const act = ev?.eval_activity, fall = ev?.eval_anomaly, resp = ev?.eval_respiration, rep = ev?.sensefi_reported as Record<string, number> | undefined;
  const allResp = resp?.groups?.All;
  const rerun = rep ? Object.entries(rep).map(([model, published]) => ({ model, published: published * 100, ours: act?.all_archs?.[model] != null ? act.all_archs[model] * 100 : null }))
    .sort((a, b) => (b.ours != null ? 1 : 0) - (a.ours != null ? 1 : 0) || b.published - a.published) : [];
  const respGroups = resp ? Object.entries(resp.groups as Record<string, any>).filter(([g]) => g !== "All") : [];

  // how uncertain each headline number is, computed from the result files
  const actCorrect = act?.confusion ? (act.confusion as number[][]).reduce((s, r, i) => s + r[i], 0) : 0;
  const actTotal = act?.confusion ? (act.confusion as number[][]).reduce((s, r) => s + r.reduce((a, b) => a + b, 0), 0) : 0;
  const actCI = actTotal ? ci95(actCorrect, actTotal) : null;
  const weakest = act ? Object.entries(act.per_class as Record<string, any>).sort((a, b) => a[1].recall - b[1].recall)[0] : null;
  const fallCI = fall?.recordings ? ci95(fall.recordings.falls_caught, fall.recordings.falls) : null;
  const alarmCI = fall?.recordings ? ci95(fall.recordings.false_alarms, fall.recordings.other) : null;
  const perPerson = resp ? Object.entries(resp.groups as Record<string, any>).filter(([g]) => g.startsWith("Subject")).map(([, s]) => s.mae as number) : [];

  const results: { value: string; what: string; detail: string; range?: string }[] = [
    { value: act ? `${(act.accuracy * 100).toFixed(1)}%` : "—", what: "of activity windows labelled correctly",
      detail: act ? `${act.n} windows from ${act.n_recordings ?? "unseen"} UT-HAR recordings the model never trained on, seven activities including falls.` : "Run scripts/eval_activity.py to fill this in.",
      range: actCI && weakest ? `95% interval ${pct(actCI[0], 1)} to ${pct(actCI[1], 1)}, which is optimistic because windows overlap within a recording. Weakest activity: ${label(weakest[0]).toLowerCase()}, ${pct(weakest[1].recall, 1)}.` : undefined },
    { value: fall ? (fall.recordings ? `${fall.recordings.falls_caught} of ${fall.recordings.falls}` : `${Math.round(fall.recall * fall.n_fall)} of ${fall.n_fall}`) : "—",
      what: fall?.recordings ? "recorded falls caught" : "falls caught",
      detail: fall ? (fall.recordings
        ? `${fall.recordings.false_alarms} false alarm across ${fall.recordings.other} recordings of other activities. An alert needs ${fall.consecutive_windows ?? 1} fall windows in a row, so it arrives about 2 seconds after the fall.`
        : `With ${Math.round(fall.false_per_100 * fall.n_other / 100)} false alarm in ${fall.n_other} windows of other activities, at a ${Math.round(fall.threshold * 100)}% confidence threshold.`) : "Run scripts/eval_activity.py to fill this in.",
      range: fallCI && alarmCI ? `With only ${fall.recordings.falls} fall recordings, the 95% interval is ${pct(fallCI[0])} to ${pct(fallCI[1])} for catching a fall and ${pct(alarmCI[0], 2)} to ${pct(alarmCI[1], 1)} for a false alarm.` : undefined },
    { value: allResp ? `±${allResp.mae.toFixed(1)}` : "—", what: "breaths a minute, average error against a chest belt",
      detail: allResp ? `${allResp.gated_mae?.toFixed(1)} on the ${Math.round(allResp.coverage * 100)}% of readings clear enough to use. ${resp.n_windows} windows, each person scored by a model that never saw them.` : "The WiRe-Breath recordings are still being processed.",
      range: perPerson.length ? `By person the error runs from ${Math.min(...perPerson).toFixed(2)} to ${Math.max(...perPerson).toFixed(2)} breaths a minute (${perPerson.length} people).` : undefined },
  ];

  return (
    <div className="grid gap-5">
      <header className="max-w-3xl px-1">
        <h1 className="text-[28px] font-extrabold tracking-tight sm:text-[30px]">Atheera reads breathing and movement from the Wi-Fi already on board.</h1>
        <p className="mt-1 text-muted-foreground">No wearables, no cameras. Here is how it works, how well, and what it does not claim.</p>
      </header>
      {q.isError && <p className="text-sm text-warning">The results didn&apos;t load. Check that the backend is running on port 8010, then reload the page.</p>}

      <ol data-tour="how-steps" className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-5">
        {STEPS.map((s, i) => (
          <li key={s.title}>
            <Card className="h-full gap-2 px-5">
              <span className="flex items-center gap-2">
                <span className="grid size-9 place-items-center rounded-xl bg-primary text-primary-foreground"><s.icon weight="bold" className="size-5" /></span>
                <span className="text-xs font-bold text-muted-foreground">Step {i + 1}</span>
              </span>
              <h2 className="mt-1 text-[15.5px] font-bold">{s.title}</h2>
              <p className="text-[13px] text-muted-foreground">{s.text}</p>
            </Card>
          </li>
        ))}
      </ol>

      <section aria-label="Results" data-tour="how-results" className="grid gap-3">
        <h2 className="px-1 text-xl font-extrabold">How well it works</h2>
        <div className="grid gap-5 lg:grid-cols-3">
          {results.map((r) => (
            <Card key={r.what} className="gap-2.5 px-5">
              {q.isPending ? <Skeleton className="h-10 w-32" /> : <span className="text-[40px] font-extrabold leading-none tracking-tighter text-primary">{r.value}</span>}
              <p className="font-bold">{r.what}</p>
              <p className="text-sm text-muted-foreground">{r.detail}</p>
              {r.range && <p className="rounded-2xl bg-sunk px-3 py-2 text-[13px] text-foreground/80">{r.range}</p>}
            </Card>
          ))}
        </div>
      </section>

      <section aria-label="Models" data-tour="how-models" className="grid gap-3">
        <h2 className="px-1 text-xl font-extrabold">What is inside</h2>
        <Card className="overflow-x-auto px-5">
          <table className="w-full min-w-[640px] border-collapse text-sm">
            <thead>
              <tr className="text-start text-muted-foreground">
                <th className="border-b px-2 py-2 text-start font-bold">What it does</th>
                <th className="border-b px-2 py-2 text-start font-bold">Method</th>
                <th className="border-b px-2 py-2 text-start font-bold">Learned from data?</th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <tr key={m.task} className="align-top">
                  <td className="border-b px-2 py-3 font-bold">{m.task}</td>
                  <td className="border-b px-2 py-3 text-foreground/80">{m.model}</td>
                  <td className="border-b px-2 py-3 text-muted-foreground">{m.learned}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="pt-3 text-[13px] text-muted-foreground">
            Breathing has no learned rate on purpose: a model trained on typical breathing can quietly pull readings toward typical rates and hide the abnormal breathing this monitor exists to catch.
          </p>
        </Card>
      </section>

      <div className="grid items-start gap-5 lg:grid-cols-2">
        <Card className="px-5">
          <CardHeader className="px-0"><CardTitle className="text-base font-bold">Activity, by type</CardTitle><CardDescription>Share of each activity the model labels correctly</CardDescription></CardHeader>
          {act ? (
            <>
              <HBar data={Object.entries(act.per_class as Record<string, any>).map(([k, v]) => ({ key: k, value: v.recall * 100 })).sort((a, b) => b.value - a.value)}
                format={(v) => `${v.toFixed(1)}%`} max={100} />
              <TableView head={["Activity", "Recall", "Precision"]}
                rows={Object.entries(act.per_class as Record<string, any>).map(([k, v]) => [label(k), `${(v.recall * 100).toFixed(1)}%`, `${(v.precision * 100).toFixed(1)}%`])} />
            </>
          ) : <Skeleton className="h-60" />}
        </Card>

        <Card className="px-5">
          <CardHeader className="px-0"><CardTitle className="text-base font-bold">Where it gets confused</CardTitle><CardDescription>{act?.model ?? "Activity model"}, windows from UT-HAR recordings it never trained on</CardDescription></CardHeader>
          {act ? (
            <>
              <Confusion classes={act.classes} matrix={act.confusion} />
              <p className="text-[13px] text-muted-foreground">Each row is one real activity; numbers are the % of its windows the model gave each label. A dark diagonal means right answers.</p>
              <TableView head={["Actual \\ predicted", ...act.classes.map(label)]} rows={act.confusion.map((r: number[], i: number) => [label(act.classes[i]), ...r])} />
            </>
          ) : <Skeleton className="h-60" />}
        </Card>

        <Card className="px-5">
          <CardHeader className="px-0"><CardTitle className="text-base font-bold">We re-ran the published models</CardTitle><CardDescription>SenseFi&apos;s own weights on SenseFi&apos;s own test split. Bars are our re-run, dots are the published numbers.</CardDescription></CardHeader>
          {rerun.length ? (
            <>
              <div style={{ height: rerun.length * 34 + 8 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={rerun} layout="vertical" margin={{ top: 0, right: 56, bottom: 0, left: 0 }} barCategoryGap={8}>
                    <CartesianGrid horizontal={false} stroke="var(--border)" />
                    <XAxis type="number" hide domain={[0, 100]} />
                    <YAxis type="category" dataKey="model" width={104} tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 13 }} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--sunk)" }}
                      formatter={(v, n) => [v == null ? "not re-run" : `${Number(v).toFixed(2)}%`, n === "ours" ? "Our re-run" : "Published"]} />
                    <Bar dataKey="ours" radius={6} fill="var(--brand-2)" background={{ fill: "var(--sunk)", radius: 6 }} isAnimationActive={false}>
                      {rerun.map((r) => <Cell key={r.model} fill={r.ours == null ? "transparent" : "var(--brand-2)"} />)}
                      <LabelList dataKey="ours" position="right" formatter={(v) => (v == null ? "—" : `${Number(v).toFixed(1)}%`)} className="fill-foreground text-[13px] font-bold" />
                    </Bar>
                    <Scatter dataKey="published" fill="var(--ink)" isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[13px] text-muted-foreground">
                Our numbers match the published ones, so the models behave as their authors reported. That split lets the same recordings appear in training and testing{act?.leakage_note ? ` (${act.leakage_note})` : ""}, so it overstates accuracy.
                Atheera uses its own model, scored on whole recordings it never saw{act?.sensefi_recipe_on_heldout_recordings ? `; SenseFi's recipe scores ${(act.sensefi_recipe_on_heldout_recordings * 100).toFixed(1)}% on the same held-out recordings` : ""}.
              </p>
            </>
          ) : <Skeleton className="h-48" />}
        </Card>

        <Card className="px-5">
          <CardHeader className="px-0"><CardTitle className="text-base font-bold">Breathing rate against the chest belt</CardTitle><CardDescription>Average error in breaths a minute, lower is better</CardDescription></CardHeader>
          {resp ? (
            <>
              <HBar data={respGroups.map(([g, s]) => ({ key: g.replace(/^(Environment|Posture): /, ""), value: s.mae })).sort((a, b) => a.value - b.value)}
                format={(v) => v.toFixed(1)} />
              <p className="text-[13px] text-muted-foreground">Readings below 45% clarity are shown as unclear and never trigger alarms.</p>
              <TableView head={["Group", "Avg error", "Within ±2", "Windows"]}
                rows={Object.entries(resp.groups as Record<string, any>).map(([g, s]) => [g, s.mae.toFixed(2), `${Math.round(s.within2 * 100)}%`, s.n])} />
            </>
          ) : <Skeleton className="h-60" />}
        </Card>
      </div>

      <div data-tour="how-notes" className="grid gap-5 lg:grid-cols-2">
        {[
          { title: "What it does not claim", items: [
            "No heart rate, sleep apnoea or seizure detection. Those are future research.",
            "It is not a medical device. It supports a flight surgeon; it does not diagnose.",
            "No data from orbit exists yet. Every recording was made on Earth."] },
          { title: "Read the numbers carefully", items: [
            "Activity results come from recordings held out of training, but every UT-HAR recording shares the same people and room, so accuracy in a new room will be lower.",
            fallCI && fall?.recordings ? `Only ${fall.recordings.falls} fall recordings were tested. Catching all of them still fits a true catch rate as low as about ${pct(fallCI[0])}.` : "",
            "The breathing method was designed by comparing variants on these same recordings, so its error is probably a little optimistic. Only the clarity score is cross-checked by person.",
            "Breathing is scored on short recordings of people lying or sitting still. Whole nights and moving crew will be harder.",
            "The demo's slow-breathing event replays a real recording slowed down. No public dataset has genuinely slow breathing, and real slow breathing may look different.",
            "Breathing and activity come from different datasets and hardware, so each crew member in the demo uses one source."].filter(Boolean) },
          { title: "How the results were checked", items: [
            "Each headline number was recomputed from the raw results (the confusion matrix and the group counts) and matches this page.",
            "38 automated tests pass: known-answer breathing rates from 7 to 34 breaths a minute, pure noise gets no confidence, the alert rules, and the rest tracker.",
            "The published SenseFi score reproduces here, and this page shows why it is too high: its test windows overlap its training recordings.",
            "Everything can be rebuilt from the public datasets with the commands in the repository. The full report is docs/VALIDATION.md."] },
          { title: "Data", items: [
            "WiRe-Breath: 5 people, 9 sleeping and sitting postures, quiet and noisy rooms, with a chest-belt reference. Ge et al., IEEE JBHI 2025, CC BY 4.0.",
            "UT-HAR via SenseFi: 7 activities including falls, Intel 5300 Wi-Fi cards. Yousefi et al. 2017; benchmark and weights by Yang et al., MIT licence."] },
        ].map((b) => (
          <Card key={b.title} className="px-5">
            <CardHeader className="px-0"><CardTitle className="text-base font-bold">{b.title}</CardTitle></CardHeader>
            <ul className="grid gap-2">{b.items.map((i) => <li key={i} className="rounded-2xl bg-sunk px-3.5 py-2.5 text-sm text-foreground/80">{i}</li>)}</ul>
          </Card>
        ))}
      </div>
    </div>
  );
}
