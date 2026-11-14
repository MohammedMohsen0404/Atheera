export type Severity = "NOMINAL" | "ADVISORY" | "CAUTION" | "WARNING" | "EMERGENCY";

export interface Frame {
  t: number;
  crew_id: string;
  name: string;
  module: string;
  occupied: boolean;
  br_available: boolean;
  bpm: number | null;
  br_conf: number;
  gt_bpm: number | null;
  activity: string | null;
  activity_probs: Record<string, number> | null;
  motion: number;
  anomaly: number;
  anomaly_reasons: string[];
  scheduled: string;
  context: string;
  status: Severity;
  source: string;
  track: string;
  synthetic: boolean;
  wave: number[] | null;
  csi: string | null;
  injected: boolean;
  /** Rest during the scheduled sleep period (backend atheer/modules/rest.py); null fields outside a sleep period. */
  rest?: RestSummary;
}

export interface RestSummary {
  state: "RESTING" | "SETTLING" | "MOVING" | "UNCLEAR" | "AWAKE_PERIOD";
  sleep_period_s: number;
  rest_s: number;
  awake_s: number;
  unclear_s: number;
  movements: number;
  nearby_motion_s: number;
  rest_pct: number | null;
  coverage_pct: number | null;
  movements_per_hour: number | null;
}

export interface Alert {
  id: number;
  t: number;
  crew_id: string;
  module: string;
  severity: Severity;
  code: string;
  title: string;
  evidence: Record<string, unknown>;
  acked: boolean;
  cleared_t: number | null;
  acked_t?: number | null;
}

export interface ModuleDef {
  id: string;
  name: string;
}

export interface CrewDef {
  id: string;
  name: string;
  module: string;
}

export interface StateMsg {
  type: "state";
  running: boolean;
  t: number;
  duration: number;
  speed: number;
  scenario: string;
  modules: ModuleDef[];
  crew: CrewDef[];
  alerts: Alert[];
  timeline: { at: number; crew: string; label: string; injected: boolean }[];
  inject_kinds: string[];
}

export interface TelemetryMsg {
  type: "telemetry";
  t: number;
  duration: number;
  frames: Frame[];
  new_alerts: Alert[];
}

export const SEVERITY_ORDER: Severity[] = ["NOMINAL", "ADVISORY", "CAUTION", "WARNING", "EMERGENCY"];

export const met = (t: number) => {
  const s = Math.max(0, Math.floor(t));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

export const prettyActivity = (a: string | null) => (a ? a.replace("_", " ") : "—");

/** Context code as words that read after "while": "while asleep", "while resting". */
export const contextWords = (c?: string | null) =>
  ({ SLEEP: "asleep", REST: "resting", ACTIVE: "active", EXERCISE: "exercising" } as Record<string, string>)[c ?? ""] ?? "resting";
