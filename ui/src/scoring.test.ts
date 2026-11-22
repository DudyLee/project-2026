import assert from 'node:assert/strict';
import test from 'node:test';
import { chooseRule, daylightWeightTotal, DEFAULT_RULE, editRule, percentagesOfPeak, preferenceScores, rank, rankAngles, setDaylightWeight, validRule } from './scoring.ts';
import type { Angle, Row } from './types.ts';

function fixture(angle: Angle, power: number, daylight: number): Row {
  return {
    PVSD_Angle_deg: angle, PV_AC_kWh: power, F_UDI_fraction: daylight,
    E_mean_lux: 0, E_P10_lux: 0, F_300_fraction: 0,
  };
}

const rows = [fixture(0, .008, .4), fixture(30, .006, .64), fixture(60, .003, .8), fixture(90, .001, .6)];
const recommended = (scores: ReturnType<typeof rank>) => scores.filter(row => row.recommended).map(row => row.PVSD_Angle_deg);

test('the same prediction can favor different angles under different preferences', () => {
  assert.deepEqual(recommended(rank(rows, chooseRule(null))), []);
  assert.deepEqual(recommended(rank(rows, chooseRule('power'))), [0]);
  assert.deepEqual(recommended(rank(rows, chooseRule('daylight'))), [60]);
  assert.deepEqual(recommended(rank(rows, chooseRule('balanced'))), [30]);
});

test('custom weights change the recommendation without mutating model data', () => {
  const rule = { power: 10, udi: 100, f300: 0, p10: 0, p10_target: 100 };
  assert.deepEqual(recommended(rank(rows, chooseRule('custom', rule))), [60]);
  assert.equal(rows[0].F_UDI_fraction, .4);
  assert.equal(validRule({ ...rule, udi: 99 }), false);
  assert.deepEqual(recommended(rank(rows, chooseRule('custom', { ...rule, udi: 99 }))), []);
});

test('ties and zero denominators remain finite', () => {
  const scores = rank([fixture(0, 0, 0), fixture(90, 0, 0)], chooseRule('balanced'));
  assert.deepEqual(scores.map(row => row.score), [0, 0]);
  assert.deepEqual(recommended(scores), [0, 90]);
});

test('comparison scores include all presets and an applied valid custom rule', () => {
  const standard = preferenceScores(rows);
  assert.deepEqual(standard.map(({ preference }) => preference), ['power', 'daylight', 'balanced']);
  for (const set of standard) assert.deepEqual(set.rows, rank(rows, chooseRule(set.preference)));

  const customRule = { power: 10, udi: 100, f300: 0, p10: 0, p10_target: 100 };
  const custom = preferenceScores(rows, customRule).at(-1);
  assert.equal(custom?.preference, 'custom');
  assert.deepEqual(custom?.rows, rank(rows, chooseRule('custom', customRule)));
  assert.equal(preferenceScores(rows, { ...customRule, udi: 99 }).length, standard.length);
});

test('each prediction metric uses its own peak and handles a zero series', () => {
  assert.deepEqual(percentagesOfPeak([20, 10, 0, 5]), [100, 50, 0, 25]);
  assert.deepEqual(percentagesOfPeak([0, 0, 0, 0]), [0, 0, 0, 0]);
  assert.deepEqual(percentagesOfPeak([.004, .002, 0, .001]), [100, 50, 0, 25]);
});

test('prediction and simulation share a metric peak and keep missing references as gaps', () => {
  const predicted = [80, 40, 30, 20];
  const simulated = [100, 60, null, 20];
  assert.deepEqual(percentagesOfPeak([...predicted, ...simulated]), [80, 40, 30, 20, 100, 60, null, 20]);
  assert.deepEqual(percentagesOfPeak([0, 0, null]), [0, 0, null]);
});

test('ranking filters before sorting and reports the closest miss for a strict limit', () => {
  const order = { metric: 'PV_AC_kWh', direction: 'desc' } as const;
  const limits = [{ metric: 'F_UDI_fraction', operator: 'gte', value: .6 }] as const;
  const matches = rankAngles(rows, [...limits], order);
  assert.deepEqual(matches.eligible.map(row => row.PVSD_Angle_deg), [30, 60, 90]);
  assert.equal(matches.closest, null);

  const strict = { metric: 'F_UDI_fraction', operator: 'gt', value: .8 } as const;
  const impossible = rankAngles(rows, [strict], order);
  assert.deepEqual(impossible.eligible, []);
  assert.equal(impossible.closest?.row.PVSD_Angle_deg, 60);
  assert.deepEqual(impossible.closest?.misses, [strict]);
});

test('the default daylight formula gives 68 for 80% useful daylight and 50% coverage', () => {
  const example = { ...fixture(0, .02, .8), F_300_fraction: .5 };
  assert.equal(rank([example], chooseRule('daylight'))[0].daylightScore, 68);
});

test('daylight edits preserve the other weights while the user allocates 100 percent', () => {
  const base = { ...DEFAULT_RULE, udi: 65, f300: 5, p10: 30, p10_target: 150 };
  const over = setDaylightWeight(base, 'f300', 10);
  assert.deepEqual(over, { ...base, f300: 10 });
  assert.equal(daylightWeightTotal(over), 105);
  assert.equal(validRule(over), false);
  const balanced = setDaylightWeight(over, 'p10', 25);
  assert.deepEqual(balanced, { ...base, f300: 10, p10: 25 });
  assert.equal(validRule(balanced), true);
});

test('a sole 100 percent weight can be reduced before adding another metric', () => {
  const onlyUdi = { ...DEFAULT_RULE, udi: 100, f300: 0 };
  const under = setDaylightWeight(onlyUdi, 'udi', 80);
  assert.equal(daylightWeightTotal(under), 80);
  assert.equal(validRule(under), false);
  assert.deepEqual(setDaylightWeight(under, 'f300', 20), { ...onlyUdi, udi: 80, f300: 20 });
  const withInvalidTarget = { ...DEFAULT_RULE, p10: 20, p10_target: 0 };
  assert.deepEqual(setDaylightWeight(withInvalidTarget, 'p10', 0), DEFAULT_RULE);
});

test('custom rules cap weighted lux terms and rank only eligible angles', () => {
  const candidates = [
    { ...fixture(0, .008, .8), F_300_fraction: .9, E_P10_lux: 60, E_mean_lux: 550 },
    { ...fixture(30, .006, .7), F_300_fraction: .5, E_P10_lux: 160, E_mean_lux: 250 },
  ];
  const cutoff = { metric: 'E_P10_lux', operator: 'gte', value: 80 } as const;
  const rule = {
    power: 50, udi: 40, f300: 30, p10: 20, p10_target: 100, mean: 10, mean_target: 500, cutoffs: [cutoff],
  };
  assert.equal(validRule(rule), true);
  assert.equal(validRule({ ...rule, mean_target: null }), false);
  const scores = rank(candidates, chooseRule('custom', rule));
  assert.deepEqual(scores.map(row => row.daylightScore), [81, 68]);
  assert.deepEqual(scores.map(row => row.score), [null, 100]);
  assert.deepEqual(recommended(scores), [30]);

  const noMatch = editRule(rule, { cutoffs: [{ ...cutoff, value: 200 }] });
  assert.deepEqual(noMatch.cutoffs, [{ ...cutoff, value: 200 }]);
  assert.deepEqual(recommended(rank(candidates, chooseRule('custom', noMatch))), []);
  assert.equal(rankAngles(candidates, noMatch.cutoffs!).closest?.row.PVSD_Angle_deg, 30);
});

test('a complete set of daylight weights replaces prior terms and cutoff edits preserve weights', () => {
  const cutoff = { metric: 'E_P10_lux', operator: 'gte', value: 80 } as const;
  const base = {
    power: 50, udi: 40, f300: 30, p10: 20, p10_target: 100, mean: 10, mean_target: 500, cutoffs: [cutoff],
  };
  const next = editRule(base, { udi: 70, f300: 30 });
  assert.deepEqual([next.udi, next.f300, next.p10, next.mean], [70, 30, 0, 0]);
  assert.deepEqual(next.cutoffs, [cutoff]);
  assert.deepEqual(editRule(base, { remove_cutoffs: ['E_P10_lux'] }), { ...base, cutoffs: [] });
});
