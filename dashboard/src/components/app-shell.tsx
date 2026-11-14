"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { Moon, Pause, Play, Question, Restart, Sun } from "@/components/icons";
import { Tour } from "@/components/tour";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { post, useTelemetry } from "@/lib/telemetry";
import { TOURS } from "@/lib/tours";
import { met } from "@/lib/types";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Crew" },
  { href: "/station/", label: "Station" },
  { href: "/log/", label: "Shift log" },
  { href: "/how/", label: "How it works" },
];

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const dark = mounted && resolvedTheme === "dark";
  return (
    <Button variant="ghost" size="icon" className="bg-card shadow-soft" onClick={() => setTheme(dark ? "light" : "dark")}
      aria-label={dark ? "Switch to the light theme" : "Switch to the dark theme"}>
      {dark ? <Sun weight="bold" /> : <Moon weight="bold" />}
    </Button>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const { state, t, connected, running, alerts, crew, frames, togglePlay, setSelected } = useTelemetry();
  const open = alerts.filter((a) => !a.acked);
  const duration = state?.duration ?? 600;

  // the guided tour of whichever page is open
  const pageKey = path === "/" ? "/" : path.replace(/\/$/, "");
  const tourSteps = TOURS[pageKey] ?? [];
  const [touring, setTouring] = useState(false);
  const endTour = useCallback(() => setTouring(false), []);
  useEffect(() => setTouring(false), [pageKey]);

  // The tour opens by itself when the website is opened, once per browser tab: reloading or moving between pages
  // does not reopen it, and the question-mark button starts it again. Add ?tour=off to the address to skip it
  // (for recordings and automated tests).
  const hasTour = tourSteps.length > 0;
  const hasTourRef = useRef(hasTour);
  hasTourRef.current = hasTour;
  useEffect(() => {
    let skip = false;
    try {
      skip = new URLSearchParams(window.location.search).get("tour") === "off" || sessionStorage.getItem("atheera-tour-seen") === "1";
    } catch { /* storage is blocked: just show the tour */ }
    if (skip) return;
    const timer = window.setTimeout(() => {
      if (!hasTourRef.current) return;
      try { sessionStorage.setItem("atheera-tour-seen", "1"); } catch { /* storage is blocked */ }
      setTouring(true);
    }, 900); // let the page draw first, so the first part is outlined in its final place
    return () => window.clearTimeout(timer);
  }, []);

  // The demo shift starts by itself whenever the website opens with nothing playing (first visit, reload, or after
  // a restart), so a visitor always sees live data instead of an empty page. A paused replay stays paused.
  // ?autoplay=off or ?tour=off skip it (recordings and automated tests drive the replay themselves).
  const autoStarted = useRef(false);
  useEffect(() => {
    if (autoStarted.current || !connected || !state || state.running || Object.keys(frames).length > 0) return;
    autoStarted.current = true;
    let skip = false;
    try {
      const q = new URLSearchParams(window.location.search);
      skip = q.get("autoplay") === "off" || q.get("tour") === "off";
    } catch { /* no address to read: start */ }
    if (!skip) post("/api/scenario/start").catch(() => { /* the Play button stays available */ });
  }, [connected, state, frames]);

  // Space plays or pauses; 1 to 4 pick a crew member
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(tag)) return;
      if (e.code === "Space") { e.preventDefault(); togglePlay(); }
      const n = Number(e.key);
      if (n >= 1 && n <= crew.length) setSelected(crew[n - 1].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePlay, crew, setSelected]);

  const send = (path: string, body?: unknown) => post(path, body).catch((e) => toast.error(`The replay didn't respond: ${e.message}`));

  return (
    <div className="mx-auto max-w-[1480px] px-4 pb-10 pt-4 sm:px-6">
      <header className="mb-5 grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 lg:grid-cols-[auto_1fr_auto] lg:gap-5">
        <Link href="/" className="col-start-1 row-start-1 flex items-center gap-2.5">
          {/* The logo sits straight on the page; a lighter copy keeps it readable on the dark theme. */}
          <img src="/logo.png" alt="" width={320} height={218} className="h-11 w-auto dark:hidden" />
          <img src="/logo-dark.png" alt="" width={320} height={218} className="hidden h-11 w-auto dark:block" />
          <span className="text-xl font-extrabold tracking-tight">Atheera</span>
          <span className="hidden text-sm text-muted-foreground sm:inline">Crew watch</span>
        </Link>

        <nav aria-label="Pages" data-tour="menu" className="col-span-2 row-start-2 flex w-fit max-w-full gap-1 overflow-x-auto rounded-full bg-card p-1.5 shadow-soft lg:col-span-1 lg:col-start-2 lg:row-start-1 lg:justify-self-center">
          {NAV.map((n) => {
            const on = n.href === "/" ? path === "/" : path.startsWith(n.href.replace(/\/$/, ""));
            return (
              <Link key={n.href} href={n.href} aria-current={on ? "page" : undefined}
                className={cn("inline-flex items-center gap-2 whitespace-nowrap rounded-full px-4 py-2 text-sm font-semibold text-muted-foreground transition-colors hover:text-foreground",
                  on && "bg-primary text-primary-foreground hover:text-primary-foreground")}>
                {n.label}
                {n.href === "/log/" && open.length > 0 && (
                  <span className="grid h-5 min-w-5 place-items-center rounded-full bg-emergency px-1.5 text-[11px] font-extrabold text-white" aria-label={`${open.length} open alerts`}>{open.length}</span>
                )}
              </Link>
            );
          })}
        </nav>

        <div data-tour="replay" className="col-start-2 row-start-1 flex items-center gap-2 justify-self-end lg:col-start-3">
          <Button size="icon" className="size-11 bg-ink text-background hover:bg-primary" onClick={togglePlay} disabled={!connected}
            aria-label={running ? "Pause the replay (Space)" : "Play the replay (Space)"}>
            {running ? <Pause weight="fill" /> : <Play weight="fill" />}
          </Button>
          <div className="grid min-w-[120px] gap-1.5 rounded-full bg-card px-4 py-2 shadow-soft">
            <span className="text-sm font-extrabold leading-none tabular-nums">T+{met(t)}</span>
            <span className="h-1 overflow-hidden rounded-full bg-sunk" aria-hidden="true">
              <span className="block h-full rounded-full bg-brand-2" style={{ width: `${Math.min(100, (t / duration) * 100)}%` }} />
            </span>
          </div>
          <select value={state?.speed ?? 4} disabled={!connected} aria-label="Replay speed"
            onChange={(e) => send("/api/scenario/speed", { speed: Number(e.target.value) })}
            className="hidden h-11 rounded-full bg-card px-4 text-sm font-semibold shadow-soft outline-none focus-visible:ring-3 focus-visible:ring-ring/50 sm:block">
            {[1, 2, 4, 8, 16].map((s) => <option key={s} value={s}>{s}× speed</option>)}
          </select>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon" className="hidden bg-card shadow-soft sm:inline-flex" onClick={() => send("/api/scenario/reset").then(() => send("/api/scenario/start"))}
                disabled={!connected} aria-label="Restart the replay"><Restart weight="bold" /></Button>
            </TooltipTrigger>
            <TooltipContent>Restart the replay</TooltipContent>
          </Tooltip>
          {tourSteps.length > 0 && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="bg-card shadow-soft" onClick={() => setTouring(true)} aria-label="Take a guided tour of this page">
                  <Question weight="bold" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Tour this page</TooltipContent>
            </Tooltip>
          )}
          <ThemeToggle />
          <span className={cn("hidden items-center gap-1.5 text-xs font-semibold md:inline-flex", connected ? "text-muted-foreground" : "text-warning")}
            title={connected ? "Connected to the Atheera backend" : "Trying to reach the backend on port 8010"}>
            <i className={cn("size-2 rounded-full", connected ? "bg-nominal" : "bg-warning")} />{connected ? "Live" : "Offline"}
          </span>
        </div>
      </header>

      <main className="min-w-0">{children}</main>
      {touring && <Tour key={pageKey} steps={tourSteps} onClose={endTour} />}
    </div>
  );
}
