import type { CopyKey } from './i18n';
import type { MetricId } from './types';

export type Metric = {
  key: MetricId;
  label: CopyKey;
  unit: string;
  scale: number;
  digits: number;
  color: string;
};

export const DAYLIGHT = '#087665';
export const SOLAR_ENERGY = '#AC5D15';

export const POWER: Metric = {
  key: 'PV_AC_kWh', label: 'solarEnergy', unit: 'Wh', scale: 1000, digits: 2, color: SOLAR_ENERGY,
};
export const METRICS: Metric[] = [
  { key: 'E_mean_lux', label: 'mean', unit: 'lux', scale: 1, digits: 1, color: '#286F9E' },
  { key: 'E_P10_lux', label: 'p10', unit: 'lux', scale: 1, digits: 1, color: '#164A60' },
  { key: 'F_300_fraction', label: 'f300', unit: '%', scale: 100, digits: 1, color: '#44877D' },
  { key: 'F_UDI_fraction', label: 'udi', unit: '%', scale: 100, digits: 1, color: DAYLIGHT },
  POWER,
];

export const FILTER_SIGNS = { gte: '≥', gt: '>', lte: '≤', lt: '<' } as const;
