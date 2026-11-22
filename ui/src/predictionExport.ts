import { rank } from './scoring';
import type { Angle, ExportTarget, ResultMessage } from './types';

export type ExportFormat = 'csv' | 'json';

function exportPrediction({ prediction, selection }: ExportTarget, angles?: Angle[]) {
  const rows = rank(prediction.rows, selection)
    .filter(row => !angles || angles.includes(row.PVSD_Angle_deg))
    .map(row => ({
      PVSD_Angle_deg: row.PVSD_Angle_deg, E_mean_lux: row.E_mean_lux, E_P10_lux: row.E_P10_lux,
      F_300_fraction: row.F_300_fraction, F_UDI_fraction: row.F_UDI_fraction, PV_AC_kWh: row.PV_AC_kWh,
      Daylight_Score: selection.preference ? row.daylightScore : null,
      Preference_Score: selection.preference ? row.score : null,
      Recommended: row.recommended,
      Eligible: row.eligible,
    }));
  return {
    Room_ID: prediction.room.id, DOY_1based: prediction.day, Hour: prediction.hour,
    Preference: selection.preference, Model_Version: prediction.version,
    Model_SHA256: prediction.model_sha256, Weather_Source: prediction.weather_source,
    rule: selection.preference ? selection.rule : null,
    coverage: prediction.coverage,
    rows,
  };
}

function flatten({ rows, rule, coverage, ...metadata }: ReturnType<typeof exportPrediction>) {
  return rows.map(row => ({
    ...metadata, ...row, Coverage_Flags: coverage.join(';'),
    Power_Weight_pct: rule?.power, UDI_Weight_pct: rule?.udi, F300_Weight_pct: rule?.f300,
    P10_Weight_pct: rule?.p10, P10_Target_lux: rule?.p10_target,
    Mean_Weight_pct: rule?.mean ?? (rule ? 0 : null), Mean_Target_lux: rule?.mean_target,
    Daylight_Cutoffs: rule ? JSON.stringify(rule.cutoffs ?? []) : null,
  }));
}

function toCsv(rows: Record<string, unknown>[]) {
  if (!rows.length) return '';
  const line = (values: unknown[]) => values.map(value => '"' + String(value ?? '').replaceAll('"', '""') + '"').join(',');
  return '\uFEFF' + [line(Object.keys(rows[0])), ...rows.map(row => line(Object.values(row)))].join('\r\n');
}

function download(format: ExportFormat, name: string, contents: string) {
  const href = URL.createObjectURL(new Blob([contents], {
    type: format === 'json' ? 'application/json;charset=utf-8' : 'text/csv;charset=utf-8',
  }));
  const link = document.createElement('a');
  link.href = href;
  link.download = name + '.' + format;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export function downloadPrediction(format: ExportFormat, target: ExportTarget, angles: Angle[]) {
  const prediction = exportPrediction(target, angles);
  if (!prediction.rows.length) return;
  const name = ['pvsd', prediction.Room_ID, prediction.DOY_1based, prediction.Hour].join('-');
  const contents = format === 'json' ? JSON.stringify(prediction, null, 2) : toCsv(flatten(prediction));
  download(format, name, contents);
}

export function downloadPredictions(format: ExportFormat, results: ResultMessage[]) {
  if (!results.length) return;
  const predictions = results.map(result => ({
    Prediction_ID: result.id,
    ...exportPrediction({ prediction: result.data, selection: result.selection }),
  }));
  const contents = format === 'json' ? JSON.stringify({ predictions }, null, 2)
    : toCsv(predictions.flatMap(({ Prediction_ID, ...prediction }) =>
      flatten(prediction).map(row => ({ Prediction_ID, ...row }))));
  download(format, 'pvsd-predictions', contents);
}
