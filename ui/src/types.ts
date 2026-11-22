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

export type MetricId = keyof Omit<Row, 'PVSD_Angle_deg'>;

export type DaylightMetricId = Exclude<MetricId, 'PV_AC_kWh'>;

export type MetricFilter = { metric: MetricId; operator: 'gt' | 'gte' | 'lt' | 'lte'; value: number };
export type MetricSort = { metric: MetricId; direction: 'asc' | 'desc' };

export interface Prediction {
  room: Room;
  day: number;
  hour: number;
  weather_source: string;
  version: string;
  model_sha256: string;
  coverage: string[];
  rows: Row[];
  simulation_rows?: Row[];
}

export interface Rule {
  power: number;
  udi: number;
  f300: number;
  p10: number;
  p10_target: number | null;
  mean?: number;
  mean_target?: number | null;
  cutoffs?: MetricFilter[];
}

export type RuleEdit = Partial<Rule> & {
  remove_cutoffs?: DaylightMetricId[];
  clear_cutoffs?: boolean;
};

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
  action: 'clarify' | 'predict' | 'explain' | 'language' | 'rule' | 'compare' | 'rank' | 'chart' | 'export' | 'help';
  result_id?: string | null;
  result_choices?: string[];
  reference_missed?: boolean;
  against_result_id?: string | null;
  against_choices?: string[];
  compare_scenario?: boolean;
  starts_new_prediction?: boolean;
  compare_cards?: boolean;
  compare_all?: boolean;
  angles?: Angle[];
  unsupported_angles?: number[];
  preference?: Preference | null;
  rule?: RuleEdit;
  rule_issue?: 'cutoff' | null;
  metrics?: MetricId[];
  filters?: MetricFilter[];
  sort?: MetricSort | null;
  ranking_issue?: 'filter' | 'sort' | null;
  visibility?: 'only' | 'show' | 'hide' | null;
  table_open?: boolean | null;
  format?: 'csv' | 'json' | null;
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
  bookmarked?: boolean;
  inspectedAngle?: Angle | null;
  hiddenMetrics?: MetricId[];
  valuesOpen?: boolean;
};

export type ComparisonMessage = {
  id: string;
  kind: 'comparison';
  language?: string;
  left: Prediction;
  right: Prediction;
  pairs: [Angle, Angle][];
  metrics?: MetricId[];
};

export type RankingMessage = {
  id: string;
  kind: 'ranking';
  language?: string;
  data: Prediction;
  angles: Angle[];
  filters: MetricFilter[];
  sort?: MetricSort | null;
};

export type ExportTarget = {
  prediction: Prediction;
  selection: RuleSelection;
  angles?: Angle[];
  format?: 'csv' | 'json' | null;
};

export type Message =
  | { id: string; kind: 'user' | 'assistant' | 'clarify' | 'error' | 'choose-result' | 'choose-angles'; text: string; language?: string }
  | ResultMessage
  | ComparisonMessage
  | RankingMessage;

export const HOURS = Array.from({ length: 12 }, (_, index) => index + 7);
export const ANGLES: Angle[] = [0, 30, 60, 90];
export const METRIC_IDS: MetricId[] = ['E_mean_lux', 'E_P10_lux', 'F_300_fraction', 'F_UDI_fraction', 'PV_AC_kWh'];
export const REFERENCE_YEAR = 2001;

export function displayedHour(hour: number) {
  return String(hour).padStart(2, '0') + ':00';
}

export function getConditions(draft: Draft): Conditions | null {
  const { room_id, day, hour } = draft;
  return room_id && day !== null && hour !== null ? { room_id, day, hour } : null;
}
