import type { MetricFilter, MetricSort, Preference, Row, Rule, RuleEdit, RuleSelection } from './types.ts';

export const PREFERENCES = ['power', 'daylight', 'balanced', 'custom'] as const;

export const DEFAULT_RULE: Rule = { power: 50, udi: 60, f300: 40, p10: 0, p10_target: null, mean: 0, mean_target: null };
export const DAYLIGHT_WEIGHTS = ['udi', 'f300', 'p10', 'mean'] as const;
export type DaylightWeight = typeof DAYLIGHT_WEIGHTS[number];

export function chooseRule(preference: Preference | null, rule = DEFAULT_RULE): RuleSelection {
  if (preference === 'custom') return { preference, rule: { ...rule, cutoffs: [...(rule.cutoffs ?? [])] } };
  const power = preference === 'power' ? 100 : preference === 'daylight' ? 0 : 50;
  return { preference, rule: { ...DEFAULT_RULE, power } };
}

export function daylightWeightTotal({ udi, f300, p10, mean = 0 }: Rule) {
  return udi + f300 + p10 + mean;
}

function validTarget(target: number | null | undefined) {
  return target != null && Number.isFinite(target) && target > 0 && target <= 10000;
}

export function rebalanceDaylight(rule: Rule, key: DaylightWeight, weight: number, basis = rule): Rule {
  const otherKeys = DAYLIGHT_WEIGHTS.filter(other => other !== key);
  const total = otherKeys.reduce((sum, other) => sum + (basis[other] ?? 0), 0);
  const chosen = total ? weight : 100;
  const shares = otherKeys.map(other => {
    const portion = (basis[other] ?? 0) * (100 - chosen);
    return { key: other, whole: total ? Math.floor(portion / total) : 0, remainder: total ? portion % total : 0 };
  });
  const unassigned = 100 - chosen - shares.reduce((sum, share) => sum + share.whole, 0);
  // Award leftover points to the largest remainders so the rounded weights sum to 100.
  shares.sort((left, right) => right.remainder - left.remainder);
  const next = { ...rule, [key]: chosen };
  shares.forEach((share, index) => { next[share.key] = share.whole + Number(index < unassigned); });
  if (next.p10 === 0 && !validTarget(next.p10_target)) next.p10_target = null;
  if (next.mean === 0 && !validTarget(next.mean_target)) next.mean_target = null;
  return next;
}

export function validRule(rule: Rule): boolean {
  const { power, udi, f300, p10, mean = 0, p10_target, mean_target, cutoffs = [] } = rule;
  return [power, udi, f300, p10, mean].every(weight => Number.isInteger(weight) && weight >= 0 && weight <= 100)
    && daylightWeightTotal(rule) === 100
    && ((p10 === 0 && p10_target == null) || validTarget(p10_target))
    && ((mean === 0 && mean_target == null) || validTarget(mean_target))
    && cutoffs.length <= 5 && cutoffs.every(({ metric, value }) => metric !== 'PV_AC_kWh'
      && Number.isFinite(value) && value >= 0 && (!metric.endsWith('_fraction') || value <= 1));
}

export function editRule(base: Rule, patch: RuleEdit): Rule {
  const { cutoffs, remove_cutoffs, clear_cutoffs, ...weights } = patch;
  const namedTotal = (weights.udi ?? 0) + (weights.f300 ?? 0) + (weights.p10 ?? 0) + (weights.mean ?? 0);
  const nextCutoffs = (clear_cutoffs ? [] : base.cutoffs ?? []).filter(cutoff => !remove_cutoffs?.some(metric => metric === cutoff.metric))
    .filter(cutoff => !cutoffs?.some(next => next.metric === cutoff.metric && next.operator.startsWith('g') === cutoff.operator.startsWith('g')));
  return {
    ...base,
    ...(namedTotal === 100 ? { udi: 0, f300: 0, p10: 0, mean: 0 } : {}),
    ...weights,
    cutoffs: [...nextCutoffs, ...(cutoffs ?? [])],
  };
}

function weightedLux(lux: number, weight: number, target: number | null | undefined) {
  return weight && target ? weight * Math.min(lux / target, 1) : 0;
}

export function rank(rows: Row[], { preference, rule }: RuleSelection) {
  const canRecommend = preference !== null && validRule(rule);
  const daylight = rows.map(row => {
    const udiContribution = rule.udi * row.F_UDI_fraction;
    const thresholdContribution = rule.f300 * row.F_300_fraction;
    const p10Contribution = weightedLux(row.E_P10_lux, rule.p10, rule.p10_target);
    const meanContribution = weightedLux(row.E_mean_lux, rule.mean ?? 0, rule.mean_target);
    return {
      udiContribution, thresholdContribution, p10Contribution, meanContribution,
      daylightScore: udiContribution + thresholdContribution + p10Contribution + meanContribution,
      eligible: failedCutoffs(row, rule.cutoffs ?? []).length === 0,
    };
  });
  const bestDaylight = Math.max(0, ...daylight.filter(parts => parts.eligible).map(parts => parts.daylightScore));
  const bestPower = Math.max(0, ...rows.filter((_, index) => daylight[index].eligible).map(row => row.PV_AC_kWh));
  const scores = rows.map((row, index) => {
    const { eligible, daylightScore } = daylight[index];
    const daylightRelative = bestDaylight ? daylightScore / bestDaylight : 0;
    const powerRelative = bestPower ? row.PV_AC_kWh / bestPower : 0;
    return { daylightRelative, powerRelative,
      score: eligible ? daylightRelative * (100 - rule.power) + powerRelative * rule.power : null };
  });
  const best = Math.max(0, ...scores.flatMap(({ score }) => score === null ? [] : [score]));
  return rows.map((row, index) => {
    const { score } = scores[index];
    return {
      ...row, ...daylight[index], ...scores[index],
      recommended: canRecommend && score !== null && Math.abs(score - best) <= 1e-9,
    };
  });
}

export type RankedRow = ReturnType<typeof rank>[number];

export function preferenceScores(rows: Row[], customRule?: Rule) {
  return PREFERENCES.filter(preference => preference !== 'custom' || (customRule && validRule(customRule)))
    .map(preference => ({ preference, rows: rank(rows, chooseRule(preference, customRule)) }));
}

export function percentagesOfPeak(values: number[]) {
  const peak = Math.max(0, ...values);
  return values.map(value => peak ? value / peak * 100 : 0);
}

function failedCutoffs(row: Row, filters: MetricFilter[]) {
  return filters.filter(filter => {
    const actual = row[filter.metric];
    switch (filter.operator) {
      case 'gte': return actual < filter.value;
      case 'gt': return actual <= filter.value;
      case 'lte': return actual > filter.value;
      case 'lt': return actual >= filter.value;
    }
  });
}

export function rankAngles(rows: Row[], filters: MetricFilter[], sort?: MetricSort | null) {
  const assess = (row: Row) => ({ row, misses: failedCutoffs(row, filters) });
  const ranked = rows.map(assess);
  const eligible = ranked.filter(item => !item.misses.length).map(item => item.row);
  const byObjective = (left: Row, right: Row) => sort ? (sort.direction === 'asc' ? 1 : -1)
    * (left[sort.metric] - right[sort.metric]) : 0;
  if (eligible.length) return { eligible: eligible.sort(byObjective), closest: null };

  const distance = ({ row, misses }: ReturnType<typeof assess>) => misses.reduce((total, filter) => {
    const scale = Math.max(filter.value, ...rows.map(candidate => Math.abs(candidate[filter.metric])));
    return total + (scale ? Math.abs(row[filter.metric] - filter.value) / scale : 0);
  }, 0);
  // Prefer fewer failed limits, then the smallest relative shortfall across metrics.
  ranked.sort((left, right) => left.misses.length - right.misses.length || distance(left) - distance(right)
    || byObjective(left.row, right.row));
  return { eligible, closest: ranked[0] ?? null };
}
