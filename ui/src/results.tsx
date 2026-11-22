import { useEffect, useId, useRef, useState, type ComponentProps, type CSSProperties, type ReactNode } from 'react';
import { ChartNoAxesColumnIncreasing, ChevronDown, Download, MessageSquareText, Zap } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import { ConditionPickers } from './conditionPickers';
import { BookmarkButton } from './predictionNavigation';
import { downloadPrediction } from './predictionExport';
import { LineChart, SeriesSwatch, type ChartAnchor, type ChartSeries } from './lineChart';
import { PREFERENCES, percentagesOfPeak, preferenceScores, rank, rankAngles, type RankedRow } from './scoring';
import { useI18n } from './i18n';
import { DAYLIGHT, SOLAR_ENERGY, POWER, METRICS, type Metric } from './metrics';
import { CutoffMisses, CutoffTags, DaylightFormula, RulePopover } from './preferenceEditor';
import { displayedHour, type Angle, type ComparisonMessage, type Conditions, type ExportTarget, type MetricId, type Prediction, type Preference, type RankingMessage, type ResultMessage, type Room, type Row, type Rule, type RuleSelection } from './types';

const PREFERENCE_COLOR: Record<Preference, string> = {
  power: SOLAR_ENERGY,
  daylight: DAYLIGHT,
  balanced: '#7548B5',
  custom: '#414A57',
};

function metricValue(row: Row, metric: Metric, format: (value: number, digits?: number) => string) {
  return format(row[metric.key] * metric.scale, metric.digits);
}

function signedDifference(delta: number, digits: number, number: ReturnType<typeof useI18n>['number']) {
  if (Math.abs(delta) < 0.5 * 10 ** -digits) return number(0, digits);
  return (delta > 0 ? '+' : '') + number(delta, digits);
}

function ResultTable({ rows, scores, focused, daylightWeights, customRule, customSelected, open, onOpen }: {
  rows: RankedRow[];
  scores: ReturnType<typeof preferenceScores>;
  focused: Angle | null;
  daylightWeights: Rule;
  customRule?: Rule;
  customSelected: boolean;
  open: boolean;
  onOpen: (open: boolean) => void;
}) {
  const { t, number } = useI18n();
  return <details className="all-values" open={open} onToggle={event => {
    if (event.currentTarget.open !== open) onOpen(event.currentTarget.open);
  }}>
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
          <th scope="row">{row.PVSD_Angle_deg}° {row.recommended && <span className="recommended-label">{t('recommended')}</span>}
            {!row.eligible && <small className="muted">{t('doesNotQualify')}</small>}</th>
          {METRICS.map(metric => <td key={metric.key}>{metricValue(row, metric, number)}</td>)}
          <td>{number(row.daylightScore, 2)}</td>
          {scores.map(({ preference, rows }) => {
            const score = rows[index].score;
            return <td key={preference}>{score === null ? '—' : number(score, 2)}</td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>
    <div className="table-weights small"><DaylightFormula rule={daylightWeights} /></div>
    {customRule && <div className="table-weights small">
      {!customSelected && <DaylightFormula rule={customRule} custom />}
      <p>{t('customMix', { daylight: 100 - customRule.power, power: customRule.power })}</p>
      {!customSelected && <CutoffTags filters={customRule.cutoffs ?? []} />}
    </div>}
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

function ScoreSummary({ prediction, selection }: {
  prediction: Prediction; selection: RuleSelection;
}) {
  const { t, number, locale } = useI18n();
  const rows = rank(prediction.rows, selection);
  const chosen = rows.filter(row => row.recommended);
  const cutoffs = selection.rule.cutoffs ?? [];
  if (!chosen.length && cutoffs.length) {
    const { closest } = rankAngles(prediction.rows, cutoffs);
    if (closest) return <div className="score-summary cutoff-summary">
      <div><strong>{t('noRecommendation')}</strong>
        <p>{t('rankingNone', { angle: closest.row.PVSD_Angle_deg })}</p>
        <CutoffMisses closest={closest} />
      </div>
    </div>;
  }
  if (!chosen.length) return null;
  const top = chosen[0];
  const label = selection.preference ? t(selection.preference) : '';
  const values = { score: number(top.daylightScore, 2), energy: metricValue(top, POWER, number) };
  const reason = t(chosen.length > 1 ? 'tieReason' : selection.preference === 'power' ? 'powerReason'
    : selection.preference === 'daylight' ? 'daylightReason' : 'tradeoff', values);
  return <div className="score-summary"
    style={{ '--recommendation-color': selection.preference && PREFERENCE_COLOR[selection.preference] } as CSSProperties}>
    <div className="score-icon"><Zap size={17} aria-hidden="true" /></div>
    <div><strong>{label} · {chosen.map(row => row.PVSD_Angle_deg + '°').join(locale === 'zh' ? '、' : ', ')} {t(chosen.length > 1 ? 'tie' : 'recommended')}</strong>
      <p>{reason}</p></div>
  </div>;
}

type ChartInteraction = Pick<ComponentProps<typeof LineChart>,
  'angles' | 'recommended' | 'hovered' | 'pinned' | 'inspection' | 'onHover' | 'onPin' | 'onClear'>;

function MetricsChart({ data, hidden, onLegend, interaction }: {
  data: Prediction;
  hidden: ReadonlySet<MetricId>;
  onLegend: (id: MetricId) => void;
  interaction: ChartInteraction;
}) {
  const { t, number } = useI18n();
  const [compare, setCompare] = useState(false);
  const [comparisonHidden, setComparisonHidden] = useState<Set<string>>(() => new Set(
    METRICS.filter(metric => metric.key !== 'E_mean_lux').map(metric => metric.key),
  ));
  const references = new Map(data.simulation_rows?.map(row => [row.PVSD_Angle_deg, row]));
  const aligned = data.rows.map(row => references.get(row.PVSD_Angle_deg));
  const canCompare = aligned.some(Boolean);
  const comparing = canCompare && compare;
  const toggle = canCompare && <button type="button" className="button secondary chart-mode" aria-pressed={comparing}
    onClick={() => setCompare(!compare)}>{t('compareSimulation')}</button>;
  const title = t('predictionMetricsChart');

  if (!comparing) {
    const series = METRICS.map(metric => ({
      id: metric.key, label: t(metric.label), color: metric.color,
      values: percentagesOfPeak(data.rows.map(row => row[metric.key])),
      details: data.rows.map(row => metricValue(row, metric, number) + (metric.unit === '%' ? '%' : ' ' + metric.unit)),
    }));
    return <LineChart {...interaction} title={title} note={t('metricScale')} controls={toggle}
      series={series} hidden={hidden} onLegendClick={id => onLegend(id as MetricId)} />;
  }

  const compared = METRICS.filter(metric => !comparisonHidden.has(metric.key)).map(metric => {
    const predicted = data.rows.map(row => row[metric.key] * metric.scale);
    const simulated = aligned.map(row => row ? row[metric.key] * metric.scale : null);
    const values = percentagesOfPeak([...predicted, ...simulated]);
    const units = metric.unit === '%' ? '%' : ' ' + metric.unit;
    const deltaUnit = metric.unit === '%' ? t('percentagePoints') : metric.unit;
    const detail = (value: number | null) => value === null ? '—' : number(value, metric.digits) + units;
    const prediction: ChartSeries = {
      id: metric.key + '-prediction', color: metric.color, label: t('predictionSeries'),
      values: values.slice(0, predicted.length), details: predicted.map(detail),
    };
    const simulation: ChartSeries = {
      id: metric.key + '-simulation', color: metric.color, label: t('simulationSeries'), dashed: true,
      values: values.slice(predicted.length), details: simulated.map(detail),
    };
    const differences = simulated.map((value, index) => value === null ? '—'
      : signedDifference(predicted[index] - value, metric.digits, number) + ' ' + deltaUnit);
    return { metric, prediction, simulation, differences };
  });
  const legend = METRICS.map(metric => ({ id: metric.key, color: metric.color, label: t(metric.label) }));

  return <LineChart {...interaction} title={title} controls={toggle}
    series={compared.flatMap(({ prediction, simulation }) => [prediction, simulation])} legend={legend} hidden={comparisonHidden}
    onLegendClick={id => setComparisonHidden(previous => toggleSeries(previous, id))}
    note={<div className="chart-comparison-note">
      <span>{t('comparisonScale')}</span>
      <span className="chart-source-legend" role="group" aria-label={t('comparisonLegend')}>
        <span><SeriesSwatch color="#52665e" />{t('predictionSeries')}</span>
        <span><SeriesSwatch color="#52665e" dashed />{t('simulationSeries')}</span>
      </span>
    </div>}
    renderTooltip={index => <table className="comparison-tooltip-values">
      <thead><tr><th scope="col">{t('predictedValue')}</th><th scope="col">{t('predictionSeries')}</th>
        <th scope="col">{t('simulationSeries')}</th><th scope="col" aria-label={t('simulationDifference')}>{t('comparisonDifference')}</th></tr></thead>
      <tbody>{compared.map(({ metric, prediction, simulation, differences }) => <tr key={metric.key}>
        <th scope="row"><span className="comparison-tooltip-metric"><SeriesSwatch color={metric.color} />{t(metric.label)}</span></th>
        <td>{prediction.details[index]}</td><td>{simulation.details[index]}</td><td>{differences[index]}</td>
      </tr>)}</tbody>
    </table>} />;
}

export const ResultCard = observer(function ResultCard({
  result, rooms, onPreference, onChangeConditions, onEditPreference, onBookmark, onExport, onExplain,
  onInspect, onLegend, onValuesOpen, ruleDraft, onRuleChange, onDismissRule, onCancelRule, onApplyRule, busy,
}: {
  result: ResultMessage;
  rooms: Room[];
  onPreference: (preference: Preference | null) => void;
  onChangeConditions: (conditions: Conditions) => void;
  onEditPreference: () => void;
  onInspect: (angle: Angle | null) => void;
  onLegend: (id: MetricId) => void;
  onValuesOpen: (open: boolean) => void;
  ruleDraft?: Rule;
  onRuleChange: (rule: Rule) => void;
  onDismissRule: () => void;
  onCancelRule: () => void;
  onApplyRule: () => void;
  onBookmark: () => void;
  onExport: () => void;
  onExplain: () => void;
  busy: boolean;
}) {
  const { t, number, locale } = useI18n();
  const ruleId = useId();
  const customControl = useRef<HTMLSpanElement>(null);
  const editButton = useRef<HTMLButtonElement>(null);
  const [hiddenScores, setHiddenScores] = useState<Set<string>>(() => new Set());
  const hiddenMetrics = new Set(result.hiddenMetrics);
  const [hovered, setHovered] = useState<AngleCursor | null>(null);
  const [touchTooltip, setTouchTooltip] = useState<AngleCursor | null>(null);
  const { data, selection, customRule } = result;
  const rows = rank(data.rows, selection);
  const angles = rows.map(row => row.PVSD_Angle_deg);
  const recommended = new Set(rows.filter(row => row.recommended).map(row => row.PVSD_Angle_deg));
  const appliedCustom = customRule ?? (selection.preference === 'custom' ? selection.rule : undefined);
  const comparison = preferenceScores(data.rows, appliedCustom);
  const scoreSeries: ChartSeries[] = comparison.map(({ preference, rows }) => ({
    id: preference, label: t(preference), color: PREFERENCE_COLOR[preference],
    values: rows.map(row => row.score),
    details: rows.map(row => row.score === null ? t('doesNotQualify') : number(row.score, 2) + ' / 100'),
  }));
  const pinnedAngle = result.inspectedAngle ?? null;
  const tooltip = hovered ?? (touchTooltip?.angle === pinnedAngle ? touchTooltip : null);
  useEffect(() => {
    const { preference } = selection;
    if (preference) setHiddenScores(current => current.has(preference) ? toggleSeries(current, preference) : current);
  }, [selection.preference]);

  function clearAngle() {
    setHovered(null);
    setTouchTooltip(null);
    onInspect(null);
  }

  function chartInteraction(chart: ChartKey) {
    return {
      angles, recommended, hovered: hovered?.angle ?? null, pinned: pinnedAngle,
      inspection: tooltip?.chart === chart ? { ...tooltip, followPointer: Boolean(hovered) } : null,
      onHover: (angle: Angle, anchor: ChartAnchor | null) => {
        if (anchor) {
          setTouchTooltip(null);
          setHovered({ chart, angle, anchor });
        } else {
          setHovered(current => current?.chart === chart && current.angle === angle ? null : current);
        }
      },
      onPin: (angle: Angle, anchor: ChartAnchor) => {
        if (anchor.touch) setHovered(null);
        const next = pinnedAngle === angle ? null : angle;
        setTouchTooltip(anchor.touch && next !== null ? { chart, angle: next, anchor } : null);
        onInspect(next);
      },
      onClear: clearAngle,
    };
  }
  return <article className="result-card" lang={locale === "zh" ? "zh-CN" : "en"} id={result.id} aria-label={data.room.id + ' · ' + t('prediction')}
    tabIndex={-1} style={{ '--recommendation-color': selection.preference && PREFERENCE_COLOR[selection.preference] } as CSSProperties}>
    <div className="card-top">
      <div><div className="card-kicker"><ChartNoAxesColumnIncreasing size={17} /> {t('prediction')}</div>
        <ConditionPickers value={{ room_id: data.room.id, day: data.day, hour: data.hour }} rooms={rooms}
          disabled={busy} unsimulated={!data.room.simulated} onRun={onChangeConditions} />
      </div>
      <div className="result-actions">
        <button type="button" className="button secondary explain-action" onClick={onExplain} disabled={busy} aria-label={t('explain')}>
          <MessageSquareText size={16} aria-hidden="true" /><span>{t('explainShort')}</span>
        </button>
        <BookmarkButton bookmarked={Boolean(result.bookmarked)} onClick={onBookmark} />
        <button type="button" className="button secondary" onClick={onExport}><Download size={16} />{t('export')}</button>
      </div>
    </div>
    <div className="preference-strip">
      <span>{t('recommendationPreference')}</span>
      <div className="pills" role="group" aria-label={t('recommendationPreference')}>
        {PREFERENCES.filter(preference => preference !== 'custom').map(preference => <button type="button" className="pill"
          key={preference} aria-pressed={selection.preference === preference}
          onClick={() => onPreference(selection.preference === preference ? null : preference)}>{t(preference)}</button>)}
        <span ref={customControl} className="custom-preference" data-selected={selection.preference === 'custom'}>
          <button type="button" className="pill" aria-pressed={selection.preference === 'custom'} disabled={busy}
            aria-haspopup={appliedCustom ? undefined : 'dialog'} aria-expanded={appliedCustom ? undefined : Boolean(ruleDraft)}
            aria-controls={!appliedCustom && ruleDraft ? ruleId : undefined}
            onClick={() => {
              if (appliedCustom && ruleDraft) onDismissRule();
              onPreference('custom');
            }}>{t('custom')}</button>
          <button ref={editButton} type="button" className="pill custom-preference-edit" aria-label={t('editPreference')}
            aria-haspopup="dialog" aria-expanded={Boolean(ruleDraft)} aria-controls={ruleDraft ? ruleId : undefined}
            title={t('editPreference')} disabled={busy} onClick={ruleDraft ? onDismissRule : onEditPreference}>
            <ChevronDown size={15} aria-hidden="true" />
          </button>
        </span>
      </div>
    </div>
    {ruleDraft && <RulePopover id={ruleId} anchor={customControl} trigger={editButton} result={result} rule={ruleDraft}
      busy={busy} onChange={onRuleChange} onDismiss={onDismissRule} onCancel={onCancelRule} onApply={onApplyRule} />}
    <ScoreSummary prediction={data} selection={selection} />
    <CutoffTags filters={selection.rule.cutoffs ?? []} />
    <div className="charts">
      <LineChart {...chartInteraction('scores')} title={t('preferenceScoresChart')} note={t('preferenceScale')}
        series={scoreSeries} hidden={hiddenScores} active={selection.preference}
        onLegendClick={id => setHiddenScores(current => toggleSeries(current, id))} />
      <MetricsChart interaction={chartInteraction('metrics')} data={data} hidden={hiddenMetrics} onLegend={onLegend} />
    </div>
    <ResultTable rows={rows} scores={comparison} focused={hovered?.angle ?? pinnedAngle}
      open={result.valuesOpen ?? false} onOpen={onValuesOpen}
      daylightWeights={selection.rule} customSelected={selection.preference === 'custom'} customRule={comparison.some(score => score.preference === 'custom') ? appliedCustom : undefined} />
  </article>;
});

export function ExportDialog({ target, onClose }: { target: ExportTarget; onClose: () => void }) {
  const { t, date } = useI18n();
  const [angles, setAngles] = useState<Angle[]>(() => target.angles?.length ? target.angles : target.prediction.rows.map(row => row.PVSD_Angle_deg));
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
      <button type="button" className={'button ' + (target.format === 'json' ? 'primary' : 'secondary')} disabled={!angles.length} onClick={() => downloadPrediction('json', target, angles)}>{t('json')}</button>
      <button type="button" className={'button ' + (target.format === 'json' ? 'secondary' : 'primary')} disabled={!angles.length} onClick={() => downloadPrediction('csv', target, angles)}>{t('csv')}</button>
    </div>
  </dialog>;
}

export function ComparisonCard({ message }: { message: ComparisonMessage }) {
  const { date, t, number, locale } = useI18n();
  const { left, right, pairs, metrics } = message;
  const shownMetrics = metrics?.length ? METRICS.filter(metric => metrics.includes(metric.key)) : METRICS;
  const condition = (data: Prediction) => data.room.id + ' · ' + date(data.day) + ' · ' + displayedHour(data.hour);
  const sameScenario = condition(left) === condition(right);
  return <article className="result-card comparison-card" aria-label={t('comparison')} lang={locale === 'zh' ? 'zh-CN' : 'en'}>
    <h2>{t('comparison')}</h2>
    <p>{sameScenario ? condition(left) : t('comparisonConditions', { left: condition(left), right: condition(right) })}</p>
    {pairs.map(([from, to]) => {
      const first = left.rows.find(row => row.PVSD_Angle_deg === from);
      const second = right.rows.find(row => row.PVSD_Angle_deg === to);
      if (!first || !second) return null;
      return <div className="table-scroll" role="region" aria-label={from + '° → ' + to + '°'} tabIndex={0} key={from + '-' + to}>
        <table className="result-table comparison-table">
          <thead><tr><th scope="col">{t('predictedValue')}</th>
            <th scope="col">{sameScenario ? from + '°' : t('referenceAngle', { angle: from })}</th>
            <th scope="col">{sameScenario ? to + '°' : t('comparisonAngle', { angle: to })}</th>
            <th scope="col">{t('difference')}</th></tr></thead>
          <tbody>{shownMetrics.map(metric => {
            const delta = (second[metric.key] - first[metric.key]) * metric.scale;
            const signed = signedDifference(delta, metric.digits, number);
            const unit = metric.unit === '%' ? t('percentagePoints') : metric.unit;
            return <tr key={metric.key}><th scope="row">{t(metric.label)}<small>{metric.unit}</small></th>
              <td>{metricValue(first, metric, number)}</td><td>{metricValue(second, metric, number)}</td>
              <td>{signed} {unit}</td></tr>;
          })}</tbody>
        </table>
      </div>;
    })}
  </article>;
}

export function RankingCard({ message }: { message: RankingMessage }) {
  const { t, date, number, locale } = useI18n();
  const { data, angles, filters, sort } = message;
  const candidates = angles.length ? data.rows.filter(row => angles.includes(row.PVSD_Angle_deg)) : data.rows;
  const { eligible, closest } = rankAngles(candidates, filters, sort);
  const objective = METRICS.find(metric => metric.key === sort?.metric);
  const columns = METRICS.filter(metric => metric === objective || filters.some(filter => filter.metric === metric.key));
  const rows = eligible.length ? eligible : closest ? [closest.row] : [];
  const best = eligible[0];
  return <article className="result-card comparison-card ranking-card" aria-label={t('angleRanking')} lang={locale === 'zh' ? 'zh-CN' : 'en'}>
    <h2>{t('angleRanking')}</h2>
    <p>{data.room.id} · {date(data.day)} · {displayedHour(data.hour)}</p>
    <CutoffTags filters={filters} extra={objective
      ? t(sort?.direction === 'asc' ? 'rankingLowest' : 'rankingHighest', { metric: t(objective.label) }) : undefined} />
    {best && objective && <p className="ranking-summary">{t('rankingBest', {
      angles: eligible.filter(row => row[objective.key] === best[objective.key]).map(row => row.PVSD_Angle_deg + '°').join(', '),
      metric: t(objective.label),
      value: metricValue(best, objective, number) + (objective.unit === '%' ? '%' : ' ' + objective.unit),
    })}</p>}
    {!best && closest && <p className="ranking-summary">{t('rankingNone', { angle: closest.row.PVSD_Angle_deg })}</p>}
    <div className="table-scroll" tabIndex={0} role="region" aria-label={t('angleRanking')}>
      <table className="result-table comparison-table">
        <thead><tr><th scope="col">{t('angle')}</th>{columns.map(column =>
          <th key={column.key} scope="col">{t(column.label)}<small>{column.unit}</small></th>)}</tr></thead>
        <tbody>{rows.map(row => <tr key={row.PVSD_Angle_deg}>
          <th scope="row">{row.PVSD_Angle_deg}°</th>
          {columns.map(column => <td key={column.key}>{metricValue(row, column, number)}</td>)}
        </tr>)}</tbody>
      </table>
    </div>
    {closest && <CutoffMisses closest={closest} />}
  </article>;
}
