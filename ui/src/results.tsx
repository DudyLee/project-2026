import { useEffect, useRef, useState } from 'react';
import { ChartNoAxesColumnIncreasing, Download, Info, MessageSquareText, PanelRightOpen, Sun, Zap } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import { PreferenceButtons } from './controls';
import { rank, type RankedRow } from './scoring';
import { useI18n, type CopyKey } from './i18n';
import { displayedHour, type Angle, type Prediction, type Preference, type ResultMessage, type Row, type RuleSelection } from './types';

type Metric = { key: keyof Omit<Row, 'PVSD_Angle_deg'>; label: CopyKey; short: CopyKey; unit: string; scale: number; digits: number };

const DAYLIGHT: Metric[] = [
  { key: 'F_UDI_fraction', label: 'udiFull', short: 'udi', unit: '%', scale: 100, digits: 1 },
  { key: 'F_300_fraction', label: 'f300Full', short: 'f300', unit: '%', scale: 100, digits: 1 },
  { key: 'E_mean_lux', label: 'mean', short: 'mean', unit: 'lux', scale: 1, digits: 1 },
  { key: 'E_P10_lux', label: 'p10', short: 'p10', unit: 'lux', scale: 1, digits: 1 },
];
const POWER: Metric = { key: 'PV_AC_kWh', label: 'solarEnergy', short: 'solarEnergy', unit: 'Wh', scale: 1000, digits: 2 };
const COLUMNS = [DAYLIGHT[2], DAYLIGHT[3], DAYLIGHT[1], DAYLIGHT[0], POWER];

function metricValue(row: Row, metric: Metric, format: (value: number, digits?: number) => string) {
  return format(row[metric.key] * metric.scale, metric.digits);
}

function BarChart({ rows, metric, power = false }: { rows: RankedRow[]; metric: Metric; power?: boolean }) {
  const { t, number } = useI18n();
  const max = Math.max(0, ...rows.map(row => row[metric.key] * metric.scale));
  const reference = max || 1;
  return <div className={'bar-chart' + (power ? ' power-chart' : '')} aria-label={t(metric.label)} role="group">
    <div className="chart-grid" aria-hidden="true" />
    {rows.map(row => {
      const value = row[metric.key] * metric.scale;
      return <div className={'bar-column' + (row.recommended ? ' recommended' : '')} key={row.PVSD_Angle_deg}
        aria-label={row.PVSD_Angle_deg + '°, ' + metricValue(row, metric, number) + ' ' + metric.unit + (row.recommended ? ', ' + t('recommended') : '')}>
        <div className="bar-plot">
          <div className="bar" style={{ height: value > 0 ? Math.max(2, (value / reference) * 76) + '%' : '0' }}>
            <span className="bar-value">{metricValue(row, metric, number)}</span>
          </div>
        </div>
        <div className="bar-label">{row.PVSD_Angle_deg}° {row.recommended && <span className="bar-mark" aria-label={t('recommended')}>●</span>}</div>
      </div>;
    })}
  </div>;
}

function ResultTable({ rows, hasRule }: { rows: RankedRow[]; hasRule: boolean }) {
  const { t, number } = useI18n();
  return <div className="table-scroll">
    <table className="result-table">
      <thead><tr>
        <th scope="col">{t('angle')}</th>
        {COLUMNS.map(metric => <th scope="col" key={metric.key}>{t(metric.short)}<small>{metric.unit}</small></th>)}
        {hasRule && <th scope="col">{t('recommendationScore')}<small>0–100</small></th>}
      </tr></thead>
      <tbody>{rows.map(row => <tr key={row.PVSD_Angle_deg} className={row.recommended ? 'chosen-row' : ''}>
        <th scope="row">{row.PVSD_Angle_deg}° {row.recommended && <span className="recommended-label">{t('recommended')}</span>}</th>
        {COLUMNS.map(metric => <td key={metric.key}>{metricValue(row, metric, number)}</td>)}
        {hasRule && <td>{number(row.score, 2)}</td>}
      </tr>)}</tbody>
    </table>
  </div>;
}

export function ScoreSummary({ prediction, selection, compact = false }: {
  prediction: Prediction; selection: RuleSelection; compact?: boolean;
}) {
  const { t, number, locale } = useI18n();
  const rows = rank(prediction.rows, selection);
  const chosen = rows.filter(row => row.recommended);
  if (!chosen.length) return null;
  const top = chosen[0];
  const label = selection.preference ? t(selection.preference) : '';
  const values = { score: number(top.daylightScore, 2), energy: metricValue(top, POWER, number) };
  const reason = t(chosen.length > 1 ? 'tieReason' : selection.preference === 'power' ? 'powerReason'
    : selection.preference === 'daylight' ? 'daylightReason' : 'tradeoff', values);
  return <div className={'score-summary' + (compact ? ' compact' : '')}>
    <div className="score-icon"><Zap size={17} aria-hidden="true" /></div>
    <div><strong>{label} · {chosen.map(row => row.PVSD_Angle_deg + '°').join(locale === 'zh' ? '、' : ', ')} {t(chosen.length > 1 ? 'tie' : 'recommended')}</strong>
      <p>{reason}</p></div>
  </div>;
}

export function ScoreDetails({ prediction, selection }: { prediction: Prediction; selection: RuleSelection }) {
  const { t, number } = useI18n();
  if (!selection.preference) return null;
  const rule = selection.rule;
  const rows = rank(prediction.rows, selection);
  return <details className="score-details">
    <summary>{t('viewCalculation')}</summary>
    <p className="small">{t('calculation', { udi: rule.udi, f300: rule.f300, p10: rule.p10, target: rule.p10_target, daylight: 100 - rule.power, power: rule.power })}</p>
    <div className="table-scroll"><table className="result-table detail-table">
      <thead><tr><th>{t('shortAngle')}</th><th>{t('udiContribution')}</th><th>{t('f300Contribution')}</th><th>{t('p10Contribution')}</th><th>{t('daylightScore')}</th><th>{t('powerRelative')}</th><th>{t('recommendationScore')}</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.PVSD_Angle_deg}><th>{row.PVSD_Angle_deg}°</th>
        <td>{number(row.udiContribution, 2)}</td><td>{number(row.thresholdContribution, 2)}</td>
        <td>{number(row.p10Contribution, 2)}</td><td>{number(row.daylightScore, 2)}</td>
        <td>{number(row.powerRelative * 100)}%</td><td>{number(row.score, 2)}</td></tr>)}</tbody>
    </table></div>
  </details>;
}

export const ResultCard = observer(function ResultCard({ result, onPreference, onOpenSide, onExport, onExplain, busy }: {
  result: ResultMessage;
  onPreference: (preference: Preference | null) => void;
  onOpenSide: () => void;
  onExport: () => void;
  onExplain: () => void;
  busy: boolean;
}) {
  const { t, date, locale } = useI18n();
  const [metric, setMetric] = useState(DAYLIGHT[0]);
  const { data, selection } = result;
  const rows = rank(data.rows, selection);
  const reported = selection.preference !== null;
  return <article className="result-card" lang={locale === "zh" ? "zh-CN" : "en"} id={result.id} aria-label={data.room.id + ' · ' + t('prediction')}>
    <div className="card-top">
      <div><div className="card-kicker"><ChartNoAxesColumnIncreasing size={17} /> {t('prediction')}</div>
        <h2>{data.room.id}<span>{date(data.day)} · {displayedHour(data.hour)}</span></h2></div>
      <div className="result-actions">
        <button type="button" className="button secondary" onClick={onOpenSide}><PanelRightOpen size={16} />{t('explore')}</button>
        <button type="button" className="button secondary" onClick={onExport}><Download size={16} />{t('export')}</button>
      </div>
    </div>
    <div className="preference-strip">
      <span>{t('recommendationPreference')}</span>
      <PreferenceButtons value={selection.preference} onChange={onPreference} allowNone />
    </div>
    <ScoreSummary prediction={data} selection={selection} />
    <div className="charts">
      <section className="chart-panel">
        <div className="chart-heading"><Sun size={17} className="daylight-icon" /><h3>{t('daylightMetric')}</h3>
          <select className="metric-select" aria-label={t('metricChooser')} value={metric.key}
            onChange={event => setMetric(DAYLIGHT.find(metric => metric.key === event.target.value) ?? DAYLIGHT[0])}>
            {DAYLIGHT.map(metric => <option key={metric.key} value={metric.key}>{t(metric.label)} ({metric.unit})</option>)}
          </select>
        </div>
        <BarChart rows={rows} metric={metric} />
      </section>
      <section className="chart-panel">
        <div className="chart-heading"><Zap size={17} className="power-icon" /><h3>{t('solar')}</h3><span>Wh</span></div>
        <BarChart rows={rows} metric={POWER} power />
      </section>
    </div>
    <ResultTable rows={rows} hasRule={reported} />
    <ScoreDetails prediction={data} selection={selection} />
    <footer className="card-footer">
      <span><Info size={15} />{t('simulationWeather')} · {t('modelVersion', { version: data.version })}{!data.room.simulated && ' · ' + t('noSimulation')}</span>
      <button type="button" className="text-button" onClick={onExplain} disabled={busy}><MessageSquareText size={15} />{t('explain')}</button>
    </footer>
  </article>;
});

export type ExportTarget = { prediction: Prediction; selection: RuleSelection };

function download(format: 'csv' | 'json', target: ExportTarget, angles: Angle[]) {
  const { prediction, selection } = target;
  const scoreRows = rank(prediction.rows, selection).filter(row => angles.includes(row.PVSD_Angle_deg));
  const metadata = {
    Room_ID: prediction.room.id, DOY_1based: prediction.day, Hour: prediction.hour,
    Preference: selection.preference, Model_Version: prediction.version,
    Model_SHA256: prediction.model_sha256, Weather_Source: prediction.weather_source,
  };
  const rows = scoreRows.map(row => ({
    PVSD_Angle_deg: row.PVSD_Angle_deg, E_mean_lux: row.E_mean_lux, E_P10_lux: row.E_P10_lux,
    F_300_fraction: row.F_300_fraction, F_UDI_fraction: row.F_UDI_fraction, PV_AC_kWh: row.PV_AC_kWh,
    Daylight_Score: selection.preference ? row.daylightScore : null,
    Preference_Score: selection.preference ? row.score : null,
    Recommended: row.recommended,
  }));
  const rule = selection.preference ? selection.rule : null;
  const flat = rows.map(row => ({
    ...metadata, ...row, Coverage_Flags: prediction.coverage.join(';'),
    Power_Weight_pct: rule?.power, UDI_Weight_pct: rule?.udi, F300_Weight_pct: rule?.f300,
    P10_Weight_pct: rule?.p10, P10_Target_lux: rule?.p10_target,
  }));
  const csvLine = (values: unknown[]) => values.map(value => '"' + String(value ?? '').replaceAll('"', '""') + '"').join(',');
  const text = format === 'json'
    ? JSON.stringify({ ...metadata, rule, coverage: prediction.coverage, rows }, null, 2)
    : '\uFEFF' + [csvLine(Object.keys(flat[0])), ...flat.map(row => csvLine(Object.values(row)))].join('\r\n');
  const href = URL.createObjectURL(new Blob([text], {
    type: format === 'json' ? 'application/json;charset=utf-8' : 'text/csv;charset=utf-8',
  }));
  const link = document.createElement('a');
  link.href = href;
  link.download = ['pvsd', prediction.room.id, prediction.day, prediction.hour].join('-') + '.' + format;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}

export function ExportDialog({ target, onClose }: { target: ExportTarget; onClose: () => void }) {
  const { t, date } = useI18n();
  const [angles, setAngles] = useState<Angle[]>(() => target.prediction.rows.map(row => row.PVSD_Angle_deg));
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => { dialog.current?.showModal(); }, []);
  const rows = rank(target.prediction.rows, target.selection);
  return <dialog className="export-dialog" ref={dialog} onClose={onClose} aria-labelledby="export-title">
    <h2 id="export-title">{t('exportTitle')}</h2>
    <p>{target.prediction.room.id} · {date(target.prediction.day)} {displayedHour(target.prediction.hour)}</p>
    <fieldset><legend>{t('chooseAngles')}</legend>
      {rows.map(row => <label key={row.PVSD_Angle_deg}>
        <input type="checkbox" checked={angles.includes(row.PVSD_Angle_deg)}
          onChange={event => setAngles(event.target.checked ? [...angles, row.PVSD_Angle_deg] : angles.filter(angle => angle !== row.PVSD_Angle_deg))} />
        {row.PVSD_Angle_deg}° {row.recommended && <small>{t('recommended')}</small>}
      </label>)}
    </fieldset>
    <div className="dialog-actions">
      <button type="button" className="button secondary" onClick={() => dialog.current?.close()}>{t('close')}</button>
      <button type="button" className="button secondary" disabled={!angles.length} onClick={() => download('json', target, angles)}>{t('json')}</button>
      <button type="button" className="button primary" disabled={!angles.length} onClick={() => download('csv', target, angles)}>{t('csv')}</button>
    </div>
  </dialog>;
}

export function SideResults({ prediction, selection }: { prediction: Prediction; selection: RuleSelection }) {
  const { t, number } = useI18n();
  const rows = rank(prediction.rows, selection);
  return <div className="side-results">
    <ScoreSummary prediction={prediction} selection={selection} compact />
    <div className="table-scroll"><table className="result-table side-table">
      <thead><tr><th>{t('shortAngle')}</th><th>{t('daylightScore')}<small>0–100</small></th><th>{t('solarEnergy')}<small>Wh</small></th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.PVSD_Angle_deg} className={row.recommended ? 'chosen-row' : ''}>
        <th scope="row">{row.PVSD_Angle_deg}°{row.recommended && <span className="recommended-label">{t('recommended')}</span>}</th>
        <td>{number(row.daylightScore, 2)}</td><td>{metricValue(row, POWER, number)}</td>
      </tr>)}</tbody>
    </table></div>
    <ScoreDetails prediction={prediction} selection={selection} />
  </div>;
}
