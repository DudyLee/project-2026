import assert from 'node:assert/strict';
import test from 'node:test';
import { tooltipPosition } from './tooltipPlacement.ts';

const desktop = { width: 1440, height: 950 };
const tooltip = { width: 226, height: 128 };

test('uses a constant offset as the cursor moves across a plot', () => {
  assert.deepEqual(tooltipPosition({ x: 365, y: 370 }, tooltip, desktop), { x: 377, y: 376 });
  assert.deepEqual(tooltipPosition({ x: 385, y: 390 }, tooltip, desktop), { x: 397, y: 396 });
});

test('clamps only at the viewport edges', () => {
  assert.deepEqual(tooltipPosition({ x: 1000, y: 360 }, tooltip, desktop), { x: 1012, y: 366 });
  assert.deepEqual(tooltipPosition({ x: 1430, y: 930 }, tooltip, desktop), { x: 1206, y: 814 });
});

test('keeps the tooltip on screen on a narrow touch viewport', () => {
  assert.deepEqual(tooltipPosition({ x: 240, y: 525 }, tooltip, { width: 390, height: 844 }),
    { x: 156, y: 531 });
});
