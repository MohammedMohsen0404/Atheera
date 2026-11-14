import { prettyActivity, type Frame, type Severity } from "./types";

/** Expected breathing range (br/min) for each scheduled context. */
export const LIMITS: Record<string, [number, number]> = { SLEEP: [8, 22], REST: [8, 25], ACTIVE: [8, 32], EXERCISE: [10, 40] };
/** Below this clarity a breathing reading is shown as unclear and never raises an alarm. */
export const CONF_MIN = 0.45;
export const ROLE: Record<string, string> = {
  CDR: "Commander", FE1: "Flight engineer", FE2: "Flight engineer", FE3: "Flight engineer", FE4: "Flight engineer",
  MS1: "Mission specialist", MS2: "Mission specialist",
};

export const STATUS_WORD: Record<Severity, string> = {
  NOMINAL: "Normal", ADVISORY: "Advisory", CAUTION: "Caution", WARNING: "Warning", EMERGENCY: "Emergency",
};

export const sevClass = (s?: Severity | string) => `sev-${(s ?? "NOMINAL").toLowerCase()}`;
export const placeName = (modules: Record<string, string>, id: string) => (modules[id] ?? id).replace(/ \(.*\)/, "");
export const firstName = (name: string) => name;

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const ACTING: Record<string, string> = {
  lie_down: "lying down", sit_down: "sitting down", stand_up: "standing up", pickup: "picking up", walk: "walking", run: "running",
};

/** What the crew member is doing, as a short phrase: "asleep", "walking". */
export function doing(f: Frame) {
  if (f.activity === "fall") return "possible fall";
  if (f.context === "SLEEP") return "asleep";
  if (f.activity) return ACTING[f.activity] ?? prettyActivity(f.activity);
  return ({ REST: "resting", ACTIVE: "working", EXERCISE: "exercising" } as Record<string, string>)[f.context] ?? "resting";
}

export interface Reading {
  /** breathing: a rate is measured here; activity: this room only has an activity model */
  kind: "breathing" | "activity";
  value: string;
  unit: string;
  reliable: boolean;
  /** one plain sentence a flight surgeon can act on */
  line: string;
}

export function reading(f: Frame | undefined, waveLen = 0): Reading {
  if (!f) return { kind: "breathing", value: "—", unit: "", reliable: false, line: "Waiting for the first reading" };
  const breath = f.br_available !== false && (f.bpm != null || waveLen > 0);
  if (!breath) {
    const act = capital(doing(f));
    const line = f.activity === "fall" ? "Fall pattern in the Wi-Fi signal"
      : f.anomaly > 0.8 ? "Moving far more than usual"
      : f.motion > 0.03 ? "Moving, no breathing sensor in this room" : "Still, no breathing sensor in this room";
    return { kind: "activity", value: act, unit: "", reliable: true, line };
  }
  const [lo, hi] = LIMITS[f.context] ?? LIMITS.REST;
  const reliable = f.bpm != null && f.br_conf >= CONF_MIN;
  if (!reliable) return { kind: "breathing", value: "—", unit: "br/min", reliable, line: "Rhythm unclear, not used for alarms" };
  const v = f.bpm!;
  const line = v < lo ? `Breathing slower than expected while ${doing(f)}`
    : v > hi ? `Breathing faster than expected while ${doing(f)}`
    : `Breathing normally while ${doing(f)}`;
  return { kind: "breathing", value: v.toFixed(0), unit: "br/min", reliable, line };
}
