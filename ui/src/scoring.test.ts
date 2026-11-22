import assert from 'node:assert/strict';
import test from 'node:test';
import { chooseRule, rank, validRule } from './scoring.ts';
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
