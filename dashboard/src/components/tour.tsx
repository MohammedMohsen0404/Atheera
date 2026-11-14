"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { Button } from "@/components/ui/button";
import type { TourStep } from "@/lib/tours";

const PAD = 8; // space between the highlighted part and its outline
const GAP = 14; // space between the outline and the explanation card
const MARGIN = 12; // space kept free at the screen edges

interface Box { top: number; left: number; width: number; height: number }

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(v, Math.max(lo, hi)));
const find = (step?: TourStep) => (step?.target ? document.querySelector<HTMLElement>(step.target) : null);

/**
 * A guided tour: dims the page, outlines one part at a time and explains it.
 * Arrow keys or the buttons move between steps, Esc leaves. The outline follows the part if the page scrolls or resizes.
 */
export function Tour({ steps, onClose }: { steps: TourStep[]; onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const [settled, setSettled] = useState(false); // true once the highlighted part has stopped moving
  const card = useRef<HTMLDivElement>(null);
  const forwardBtn = useRef<HTMLButtonElement>(null);
  const direction = useRef(1);
  const step = steps[index];
  const last = index === steps.length - 1;

  const go = useCallback((delta: 1 | -1) => {
    direction.current = delta;
    setIndex((i) => {
      const n = i + delta;
      if (n < 0) return 0;
      return n;
    });
  }, []);

  const forward = useCallback(() => {
    if (index >= steps.length - 1) onClose();
    else go(1);
  }, [index, steps.length, onClose, go]);

  // keyboard: arrows move, Esc leaves
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); forward(); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [forward, go, onClose]);

  // give the focus back to where it was when the tour ends
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    return () => before?.focus?.();
  }, []);

  // each step: skip it if its part is not on this page, otherwise bring the part into view and follow it
  useEffect(() => {
    if (step?.target && !find(step)) {
      const n = index + direction.current;
      if (n >= 0 && n < steps.length) setIndex(n); else onClose();
      return;
    }
    forwardBtn.current?.focus({ preventScroll: true });
    const el = find(step);
    if (!el) { setBox(null); setSettled(true); return; }
    setSettled(false); // the card stays hidden while the page scrolls, then appears next to the part's final position
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const r0 = el.getBoundingClientRect();
    const vh = window.innerHeight;
    const ch = card.current?.offsetHeight ?? 240;
    // when the part and the card fit on screen together, scroll so the card gets room below the part;
    // when the part is too big for that, show its top and let the card sit over its lower edge
    const fits = r0.height + PAD * 2 + GAP + ch <= vh - MARGIN * 2;
    const wantTop = fits ? Math.max(MARGIN + PAD, (vh - (r0.height + GAP + ch)) / 2) : MARGIN + PAD;
    window.scrollBy({ top: r0.top - wantTop, behavior: reduce ? "auto" : "smooth" });
    let raf = 0;
    let last: Box | null = null;
    let timer: number | undefined;
    const settleSoon = () => { window.clearTimeout(timer); timer = window.setTimeout(() => setSettled(true), 180); };
    settleSoon(); // in case nothing moves at all
    const loop = () => {
      const r = el.getBoundingClientRect();
      const b = { top: r.top, left: r.left, width: r.width, height: r.height };
      if (!last || Math.abs(b.top - last.top) > 0.5 || Math.abs(b.left - last.left) > 0.5
        || Math.abs(b.width - last.width) > 0.5 || Math.abs(b.height - last.height) > 0.5) {
        last = b;
        setBox(b);
        setSettled(false);
        settleSoon();
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); window.clearTimeout(timer); };
  }, [index, step, steps.length, onClose]);

  // put the explanation next to the highlighted part: below, above, beside, or at the bottom of the screen
  useLayoutEffect(() => {
    const vw = window.innerWidth, vh = window.innerHeight;
    const cw = card.current?.offsetWidth ?? 360, ch = card.current?.offsetHeight ?? 200;
    const sx = (x: number) => clamp(x, MARGIN, vw - cw - MARGIN);
    const sy = (y: number) => clamp(y, MARGIN, vh - ch - MARGIN);
    if (!step?.target || !box) { setPos({ top: sy((vh - ch) / 2), left: sx((vw - cw) / 2) }); return; }
    const below = vh - (box.top + box.height + PAD) - MARGIN;
    const above = box.top - PAD - MARGIN;
    const right = vw - (box.left + box.width + PAD) - MARGIN;
    const left = box.left - PAD - MARGIN;
    const midX = box.left + box.width / 2 - cw / 2, midY = box.top + box.height / 2 - ch / 2;
    if (below >= ch + GAP) setPos({ top: box.top + box.height + PAD + GAP, left: sx(midX) });
    else if (above >= ch + GAP) setPos({ top: box.top - PAD - GAP - ch, left: sx(midX) });
    else if (right >= cw + GAP) setPos({ top: sy(midY), left: box.left + box.width + PAD + GAP });
    else if (left >= cw + GAP) setPos({ top: sy(midY), left: box.left - PAD - GAP - cw });
    else setPos({ top: vh - ch - MARGIN * 2, left: sx((vw - cw) / 2) }); // a large part: sit over its lower edge
  }, [box, step, index]);

  if (!step) return null;
  return createPortal(
    <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-label="Guided tour">
      {/* catches clicks on the dimmed page, so nothing underneath changes while the tour runs */}
      <div className="absolute inset-0" />
      {box && step.target ? (
        <div aria-hidden="true" className="pointer-events-none absolute rounded-2xl motion-safe:transition-[top,left,width,height] motion-safe:duration-200"
          style={{ top: box.top - PAD, left: box.left - PAD, width: box.width + PAD * 2, height: box.height + PAD * 2,
            boxShadow: "0 0 0 9999px rgba(6, 14, 30, 0.62)", outline: "2px solid rgba(255, 255, 255, 0.9)" }} />
      ) : (
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-[rgba(6,14,30,0.62)]" />
      )}
      <div ref={card} className="absolute w-[min(360px,calc(100vw-24px))] rounded-3xl bg-card p-5 text-card-foreground shadow-float motion-safe:transition-opacity motion-safe:duration-150"
        style={pos ? { top: pos.top, left: pos.left, opacity: settled ? 1 : 0 } : { visibility: "hidden" }}>
        <p className="text-xs font-bold text-muted-foreground">{index + 1} of {steps.length}</p>
        <h2 className="mt-1 text-lg font-extrabold tracking-tight">{step.title}</h2>
        <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{step.text}</p>
        <div className="mt-4 flex items-center justify-between gap-3">
          <button type="button" onClick={onClose} className="rounded-md text-sm font-semibold text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50">
            Skip tour
          </button>
          <div className="flex gap-2">
            {index > 0 && <Button variant="outline" size="sm" onClick={() => go(-1)}>Back</Button>}
            <Button ref={forwardBtn} size="sm" onClick={forward}>{last ? "Done" : "Next"}</Button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
