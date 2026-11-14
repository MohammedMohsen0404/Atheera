"use client";

import type { RestSummary } from "@/lib/types";

/** Seconds as minutes for a short shift, hours once the period is long. */
const dur = (s: number) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : `${(s / 60).toFixed(1)} min`);

/** Compact rest summary for a crew member in their sleep period; sits inside the Activity card so the layout
 *  keeps its size. Rest = steady breathing and no movement for 30 s (backend atheer/modules/rest.py). */
export function RestLine({ r }: { r?: RestSummary }) {
  if (!r || r.sleep_period_s <= 0) return null;
  const total = r.sleep_period_s;
  const part = (s: number) => `${(100 * s) / total}%`;
  const detail = [
    `Asleep is from tonight's schedule. Rest means steady breathing and no movement for 30 s; it is rest, not sleep stages.`,
    `Awake or moving ${dur(r.awake_s)}.`,
    r.unclear_s > 0 ? `Signal unclear ${dur(r.unclear_s)}, not counted as rest.` : "",
    r.nearby_motion_s > 0 ? `Movement nearby for ${Math.round(r.nearby_motion_s)} s while breathing stayed steady.` : "",
  ].filter(Boolean).join(" ");
  return (
    <div data-tour="rest" className="grid gap-1.5" title={detail}>
      <div className="flex h-2 overflow-hidden rounded-full bg-sunk" role="img"
        aria-label={`Resting ${dur(r.rest_s)} of ${dur(total)} in the sleep period; awake ${dur(r.awake_s)}; unclear ${dur(r.unclear_s)}`}>
        <span className="h-full bg-brand" style={{ width: part(r.rest_s) }} />
        <span className="h-full bg-caution" style={{ width: part(r.awake_s) }} />
      </div>
      <p className="text-sm text-muted-foreground">
        <b className="text-foreground">Resting {dur(r.rest_s)}</b> of {dur(total)} sleep period · {r.movements} {r.movements === 1 ? "movement" : "movements"}
      </p>
    </div>
  );
}
