import type { Preference, Row, Rule, RuleSelection } from './types.ts';

export const PREFERENCES = ['power', 'daylight', 'balanced', 'custom'] as const;

export const DEFAULT_RULE: Rule = { power: 50, udi: 50, f300: 30, p10: 20, p10_target: 100 };

export function chooseRule(preference: Preference | null, rule = DEFAULT_RULE): RuleSelection {
  if (preference === 'custom') return { preference, rule: { ...rule } };
  const power = preference === 'power' ? 100 : preference === 'daylight' ? 0 : 50;
  return { preference, rule: { ...DEFAULT_RULE, power } };
}

export function validRule(rule: Rule): boolean {
  const weights = [rule.udi, rule.f300, rule.p10, rule.power];
  return weights.every(weight => Number.isInteger(weight) && weight >= 0 && weight <= 100)
    && rule.udi + rule.f300 + rule.p10 === 100
    && Number.isInteger(rule.p10_target) && rule.p10_target >= 1 && rule.p10_target <= 10000;
}

export function rank(rows: Row[], { preference, rule }: RuleSelection) {
  const canRecommend = preference !== null && validRule(rule);
  const daylight = rows.map(row => ({
    udiContribution: rule.udi * row.F_UDI_fraction,
    thresholdContribution: rule.f300 * row.F_300_fraction,
    p10Contribution: rule.p10 * Math.min(row.E_P10_lux / Math.max(rule.p10_target, 1), 1),
  }));
  const daylightScores = daylight.map(parts => parts.udiContribution + parts.thresholdContribution + parts.p10Contribution);
  const bestDaylight = Math.max(0, ...daylightScores);
  const bestPower = Math.max(0, ...rows.map(row => row.PV_AC_kWh));
  const scores = rows.map((row, index) => {
    const daylightRelative = bestDaylight ? daylightScores[index] / bestDaylight : 0;
    const powerRelative = bestPower ? row.PV_AC_kWh / bestPower : 0;
    return { daylightRelative, powerRelative, score: daylightRelative * (100 - rule.power) + powerRelative * rule.power };
  });
  const best = Math.max(0, ...scores.map(parts => parts.score));
  return rows.map((row, index) => ({
    ...row, ...daylight[index], ...scores[index], daylightScore: daylightScores[index],
    recommended: canRecommend && Math.abs(scores[index].score - best) <= 1e-9,
  }));
}

export type RankedRow = ReturnType<typeof rank>[number];
