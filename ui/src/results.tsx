import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { ChartNoAxesColumnIncreasing, Download, Info, MessageSquareText, PanelRightOpen, Zap } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import { PreferenceButtons } from './controls';
import { LineChart, type ChartAnchor, type ChartSeries } from './lineChart';
import { percentagesOfPeak, preferenceScores, rank, type RankedRow } from './scoring';
import { useI18n, type CopyKey } from './i18n';
import { displayedHour, type Angle, type Prediction, type Preference, type ResultMessage, type Row, type Rule, type RuleSelection } from './types';

type Metric = {
  key: keyof Omit<Row, 'PVSD_Angle_deg'>;
  label: CopyKey;
  unit: string;
  scale: number;
  digits: number;
  color: string;
};

const DAYLIGHT = '#087665';
const SOLAR_ENERGY = '#AC5D15';

const POWER: Metric = {
  key: 'PV_AC_kWh', label: 'solarEnergy', unit: 'Wh', scale: 1000, digits: 2, color: SOLAR_ENERGY,
};
const METRICS: Metric[] = [
  { key: 'E_mean_lux', label: 'mean', unit: 'lux', scale: 1, digits: 1, color: '#286F9E' },
  { key: 'E_P10_lux', label: 'p10', unit: 'lux', scale: 1, digits: 1, color: '#164A60' },
  { key: 'F_300_fraction', label: 'f300', unit: '%', scale: 100, digits: 1, color: '#44877D' },
  { key: 'F_UDI_fraction', label: 'udi', unit: '%', scale: 100, digits: 1, color: DAYLIGHT },
  POWER,
];

const PREFERENCE_COLOR: Record<Preference, string> = {
  power: SOLAR_ENERGY,
  daylight: DAYLIGHT,
  balanced: '#7548B5',
  custom: '#414A57',
};

function metricValue(row: Row, metric: Metric, format: (value: number, digits?: number) => string) {
  return format(row[metric.key] * metric.scale, metric.digits);
}

function ResultTable({ rows, scores, focused, daylightWeights, customRule }: {
  rows: RankedRow[];
  scores: ReturnType<typeof preferenceScores>;
  focused: Angle | null;
  daylightWeights: Rule;
  customRule?: Rule;
}) {
  const { t, number } = useI18n();
  return <details className="all-values">
    <summary>{t('viewValues')}</summary>
    <div className="table-scroll all-values-scroll" role="region" aria-label={t('viewValues')} tabIndex={0}>
      <table className="result-table">
        <colgroup span={1} />
        <colgroup span={METRICS.length} />
        <colgroup span={scores.length + 1} />
        <thead>
          <tr className="table-groups">
            <th scope="col" rowSpan={2}>{t('angle')}</th>
            <th scope="colgroup" colSpan={METRICS.length}>{t('rawPredictions')}</th>
            <th scope="colgroup" colSpan={scores.length + 1}>{t('derivedScores')}</th>
          </tr>
          <tr>
            {METRICS.map(metric => <th scope="col" key={metric.key}>{t(metric.label)}<small>{metric.unit}</small></th>)}
            <th scope="col">{t('absoluteDaylightScore')}<small>0–100</small></th>
            {scores.map(({ preference }) => <th scope="col" key={preference}>{t(preference)}<small>0–100</small></th>)}
          </tr>
        </thead>
        <tbody>{rows.map((row, index) => <tr key={row.PVSD_Angle_deg}
          className={row.recommended ? 'chosen-row' : ''} data-highlighted={row.PVSD_Angle_deg === focused}>
          <th scope="row">{row.PVSD_Angle_deg}° {row.recommended && <span className="recommended-label">{t('recommended')}</span>}</th>
          {METRICS.map(metric => <td key={metric.key}>{metricValue(row, metric, number)}</td>)}
          <td>{number(row.daylightScore, 2)}</td>
          {scores.map(({ preference, rows }) => <td key={preference}>{number(rows[index].score, 2)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
    <p className="table-weights small">{t('daylightWeights', {
      udi: daylightWeights.udi, f300: daylightWeights.f300, p10: daylightWeights.p10, target: daylightWeights.p10_target,
    })}</p>
    {customRule && <p className="table-weights small">{t('customWeights', {
      daylight: 100 - customRule.power, power: customRule.power, udi: customRule.udi, f300: customRule.f300,
      p10: customRule.p10, target: customRule.p10_target,
    })}</p>}
  </details>;
}

type ChartKey = 'scores' | 'metrics';
type AngleCursor = { chart: ChartKey; angle: Angle; anchor: ChartAnchor };

function toggleSeries(previous: Set<string>, seriesId: string) {
  const next = new Set(previous);
  if (next.has(seriesId)) next.delete(seriesId);
  else next.add(seriesId);
  return next;
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
  return <div className={'score-summary' + (compact ? ' compact' : '')}
    style={{ '--recommendation-color': selection.preference && PREFERENCE_COLOR[selection.preference] } as CSSProperties}>
    <div className="score-icon"><Zap size={17} aria-hidden="true" /></div>
    <div><strong>{label} · {chosen.map(row => row.PVSD_Angle_deg + '°').join(locale === 'zh' ? '、' : ', ')} {t(chosen.length > 1 ? 'tie' : 'recommended')}</strong>
      <p>{reason}</p></div>
  </div>;
}

export const ResultCard = observer(function ResultCard({ result, onPreference, onOpenSide, onExport, onExplain, busy }: {
  result: ResultMessage;
  onPreference: (preference: Preference | null) => void;
  onOpenSide: () => void;
  onExport: () => void;
  onExplain: () => void;
  busy: boolean;
}) {
  const { t, date, number, locale } = useI18n();
  const [hiddenScores, setHiddenScores] = useState<Set<string>>(() => new Set());
  const [hiddenMetrics, setHiddenMetrics] = useState<Set<string>>(() => new Set());
  const [hovered, setHovered] = useState<AngleCursor | null>(null);
  const [pinned, setPinned] = useState<AngleCursor | null>(null);
  const { data, selection, customRule } = result;
  const rows = rank(data.rows, selection);
  const angles = rows.map(row => row.PVSD_Angle_deg);
  const recommended = new Set(rows.filter(row => row.recommended).map(row => row.PVSD_Angle_deg));
  const appliedCustom = customRule ?? (selection.preference === 'custom' ? selection.rule : undefined);
  const comparison = preferenceScores(data.rows, appliedCustom);
  const scoreSeries: ChartSeries[] = comparison.map(({ preference, rows }) => ({
    id: preference, label: t(preference), color: PREFERENCE_COLOR[preference],
    values: rows.map(row => row.score),
    details: rows.map(row => number(row.score, 2) + ' / 100'),
  }));
  const metricSeries: ChartSeries[] = METRICS.map(metric => ({
    id: metric.key, label: t(metric.label), color: metric.color,
    values: percentagesOfPeak(data.rows.map(row => row[metric.key])),
    details: data.rows.map(row => metricValue(row, metric, number) + (metric.unit === '%' ? '%' : ' ' + metric.unit)),
  }));
  const tooltip = hovered ?? pinned;

  function choosePreference(preference: Preference | null) {
    if (preference && hiddenScores.has(preference)) {
      setHiddenScores(current => toggleSeries(current, preference));
    }
    onPreference(preference);
  }

  function clearAngle() {
    setHovered(null);
    setPinned(null);
  }

  function chartInteraction(chart: ChartKey) {
    return {
      angles, recommended, hovered: hovered?.angle ?? null, pinned: pinned?.angle ?? null,
      inspection: tooltip?.chart === chart ? { ...tooltip, followPointer: Boolean(hovered) } : null,
      onHover: (angle: Angle, anchor: ChartAnchor | null) => setHovered(current => {
        if (anchor) return { chart, angle, anchor };
        return current?.chart === chart && current.angle === angle ? null : current;
      }),
      onPin: (angle: Angle, anchor: ChartAnchor) => {
        if (anchor.touch) setHovered(null);
        setPinned(current => current?.angle === angle ? null : { chart, angle, anchor });
      },
      onClear: clearAngle,
    };
  }
  return <article className="result-card" lang={locale === "zh" ? "zh-CN" : "en"} id={result.id} aria-label={data.room.id + ' · ' + t('prediction')}
    style={{ '--recommendation-color': selection.preference && PREFERENCE_COLOR[selection.preference] } as CSSProperties}>
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
      <PreferenceButtons value={selection.preference} onChange={choosePreference} allowNone />
    </div>
    <ScoreSummary prediction={data} selection={selection} />
    <div className="charts">
      <LineChart {...chartInteraction('scores')} title={t('preferenceScoresChart')} note={t('preferenceScale')}
        series={scoreSeries} hidden={hiddenScores} active={selection.preference}
        onLegendClick={id => setHiddenScores(current => toggleSeries(current, id))} />
      <LineChart {...chartInteraction('metrics')} title={t('predictionMetricsChart')} note={t('metricScale')}
        series={metricSeries} hidden={hiddenMetrics}
        onLegendClick={id => setHiddenMetrics(current => toggleSeries(current, id))} />
    </div>
    <ResultTable rows={rows} scores={comparison} focused={pinned?.angle ?? hovered?.angle ?? null}
      daylightWeights={selection.rule} customRule={comparison.some(score => score.preference === 'custom') ? appliedCustom : undefined} />
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
  </div>;
}
