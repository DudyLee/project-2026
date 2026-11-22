import assert from 'node:assert/strict';
import test from 'node:test';
import { chooseRule, percentagesOfPeak, preferenceScores, rank, validRule } from './scoring.ts';
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
