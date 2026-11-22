export type Preference = 'power' | 'daylight' | 'balanced' | 'custom';
export type Angle = 0 | 30 | 60 | 90;

export interface Room {
  id: string;
  level: string;
  area: number;
  simulated: boolean;
}

export interface Conditions {
  room_id: string;
  day: number;
  hour: number;
}

export interface Draft {
  room_id: string | null;
  day: number | null;
  hour: number | null;
  preference: Preference | null;
}

export interface Row {
  PVSD_Angle_deg: Angle;
  E_mean_lux: number;
  E_P10_lux: number;
  F_300_fraction: number;
  F_UDI_fraction: number;
  PV_AC_kWh: number;
}

export interface Prediction {
  room: Room;
  day: number;
  hour: number;
  weather_source: string;
  version: string;
  model_sha256: string;
  coverage: string[];
  rows: Row[];
}

export interface Rule {
  power: number;
  udi: number;
  f300: number;
  p10: number;
  p10_target: number;
}

export interface RuleSelection {
  preference: Preference | null;
  rule: Rule;
}

export type Issue =
  | { field: 'room'; reason: 'missing' | 'multiple' | 'not_found'; rooms: Room[]; floor: number | null }
  | { field: 'day' | 'hour'; reason: 'missing' | 'multiple' | 'out_of_range' | 'ambiguous'; options: number[] };

export interface ChatResponse {
  draft: Draft;
  issues: Issue[];
  action: 'clarify' | 'predict' | 'explain' | 'language';
  language: string;
  requested_language: string | null;
  localized_reply: string | null;
  interpretation: 'agent' | 'local';
}

export type ResultMessage = {
  id: string;
  kind: 'result';
  language?: string;
  data: Prediction;
  selection: RuleSelection;
  customRule?: Rule;
};

export type Message =
  | { id: string; kind: 'user' | 'assistant' | 'clarify' | 'error'; text: string; language?: string }
  | ResultMessage;

export const HOURS = Array.from({ length: 12 }, (_, index) => index + 7);
export const ANGLES: Angle[] = [0, 30, 60, 90];
export const REFERENCE_YEAR = 2001;

export function displayedHour(hour: number) {
  return String(hour).padStart(2, '0') + ':00';
}

export function getConditions(draft: Draft): Conditions | null {
  const { room_id, day, hour } = draft;
  return room_id && day !== null && hour !== null ? { room_id, day, hour } : null;
}
