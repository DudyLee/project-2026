import { useId, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { useI18n } from './i18n';
import { FILTER_SIGNS, METRICS } from './metrics';
import { daylightWeightTotal, rank, rankAngles, setDaylightWeight, validRule, type RankedRow } from './scoring';
import type { MetricFilter, ResultMessage, Rule } from './types';

const TERMS = [
  { weight: 'udi', metric: 'F_UDI_fraction' },
  { weight: 'f300', metric: 'F_300_fraction' },
  { weight: 'p10', metric: 'E_P10_lux', target: 'p10_target' },
  { weight: 'mean', metric: 'E_mean_lux', target: 'mean_target' },
] as const;

export function DaylightFormula({ rule, custom = false }: { rule: Rule; custom?: boolean }) {
  const { t, number } = useI18n();
  const terms = TERMS.flatMap(term => {
    const weight = rule[term.weight] ?? 0;
    if (!weight) return [];
    const label = t(METRICS.find(metric => metric.key === term.metric)!.label);
    const target = 'target' in term ? rule[term.target] : undefined;
    return [number(weight) + ' × ' + (target ? 'min(' + label + ' / ' + number(target, 2) + ' lux, 1)' : label)];
  });

  return <p className="daylight-formula">{t(custom ? 'customDaylightScore' : 'absoluteDaylightScore')} = {terms.join(' + ')}</p>;
}

export function CutoffTags({ filters, extra, onRemove }: {
  filters: MetricFilter[]; extra?: string; onRemove?: (index: number) => void;
}) {
  const { t, number } = useI18n();
  if (!filters.length && !extra) return null;
  return <ul className="ranking-criteria">
    {filters.map((filter, index) => {
      const metric = METRICS.find(item => item.key === filter.metric)!;
      const units = metric.unit === '%' ? '%' : ' ' + metric.unit;
      const label = `${t(metric.label)} ${FILTER_SIGNS[filter.operator]} ${number(filter.value * metric.scale, 4)}${units}`;
      return <li className={onRemove ? 'editable-cutoff' : undefined} key={index}>{label}
        {onRemove && <button type="button" className="icon-button cutoff-remove" aria-label={t('removeCutoff', { label })}
          onClick={() => onRemove(index)}><X size={14} aria-hidden="true" /></button>}
      </li>;
    })}
    {extra && <li>{extra}</li>}
  </ul>;
}

export function CutoffMisses({ closest }: { closest: NonNullable<ReturnType<typeof rankAngles>['closest']> }) {
  const { t, number } = useI18n();
  return <ul className="ranking-shortfall">{closest.misses.map((filter, index) => {
    const metric = METRICS.find(item => item.key === filter.metric)!;
    const delta = (closest.row[filter.metric] - filter.value) * metric.scale;
    if (delta === 0) return <li key={index}>{t('rankingAtLimit', { metric: t(metric.label) })}</li>;
    const magnitude = Math.abs(delta) < 0.01 ? '<' + number(0.01, 2) : number(Math.abs(delta), 2);
    return <li key={index}>{t(delta < 0 ? 'rankingBelow' : 'rankingAbove', {
      metric: t(metric.label), amount: magnitude + ' ' + (metric.unit === '%' ? t('percentagePoints') : metric.unit),
    })}</li>;
  })}</ul>;
}

function RuleDetails({ rule, rows, hidden }: { rule: Rule; rows: RankedRow[]; hidden: boolean }) {
  const { t, number } = useI18n();
  const hasCutoffs = Boolean(rule.cutoffs?.length);
  return <details className="rule-preview" hidden={hidden}>
    <summary>{t('rulePreviewDetails')}</summary>
    <DaylightFormula rule={rule} />
    <p className="muted small">{t('ruleFractions')}</p>
    <table className="result-table">
      <thead><tr><th scope="col">{t('angle')}</th><th scope="col">{t('absoluteDaylightScore')}</th>
        {hasCutoffs && <th scope="col">{t('eligibility')}</th>}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.PVSD_Angle_deg}>
        <th scope="row">{row.PVSD_Angle_deg}°</th>
        <td>{number(row.daylightScore, 2)}</td>
        {hasCutoffs && <td>{t(row.eligible ? 'qualifies' : 'doesNotQualify')}</td>}
      </tr>)}</tbody>
    </table>
  </details>;
}

function PercentInput({ label, value, onChange, complement }: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  complement?: string;
}) {
  const id = useId();
  const [typed, setTyped] = useState<string>();
  return <div className="percent-control">
    <div className="percent-heading">
      {complement && <span>{complement} <strong>{100 - value}%</strong></span>}
      <label htmlFor={id}>{label}</label>
      <span className="weight-input percent-number">
        <input id={id} className="field" type="number" min={0} max={100} step={1}
          value={typed ?? value} onBlur={() => setTyped(undefined)}
          onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); }}
          onChange={event => {
            const input = event.currentTarget;
            setTyped(input.value);
            if (input.value && input.validity.valid) onChange(input.valueAsNumber);
          }} />
        <small aria-hidden="true">%</small>
      </span>
    </div>
    <input type="range" min={0} max={100} step={1} value={value}
      aria-label={label}
      onChange={event => onChange(event.currentTarget.valueAsNumber)} />
  </div>;
}

function RuleControls({ rule, onChange }: {
  rule: Rule; onChange: (rule: Rule) => void;
}) {
  const { t } = useI18n();
  const cutoffs = rule.cutoffs ?? [];
  const total = daylightWeightTotal(rule);
  const amount = Math.abs(100 - total);
  const balance = t(amount === 1 ? 'ruleWeightPoint' : 'ruleWeightPoints', { amount });
  const weightStatus = total === 100 ? t('ruleWeightAllocated')
    : t(total < 100 ? 'ruleWeightRemaining' : 'ruleWeightExcess', { percent: total, balance });

  return <>
    <div className="rule-mix">
      <PercentInput label={t('solarEnergy')} complement={t('daylightMetric')} value={rule.power}
        onChange={power => onChange({ ...rule, power })} />
    </div>
    <fieldset className="rule-weights">
      <legend>{t('daylightWeights')}</legend>
      <p className={'small ' + (total > 100 ? 'warning-text' : 'muted')} aria-live="polite">{weightStatus}</p>
      <div className="weight-grid">
        {TERMS.map(term => {
          const label = t(METRICS.find(metric => metric.key === term.metric)!.label);
          const weight = rule[term.weight] ?? 0;
          const target = 'target' in term ? term.target : undefined;
          return <div className="weight-term" key={term.weight}>
            <PercentInput label={label} value={weight} onChange={value => onChange(setDaylightWeight(rule, term.weight, value))} />
            {target && weight > 0 && <label className="lux-target">
              <span>{t('fullCreditTarget', { metric: label })}</span>
              <span className="weight-input">
                <input className="field" type="number" min={0} max={10000} step="any" value={rule[target] ?? ''}
                  onChange={event => onChange({ ...rule, [target]: event.target.value ? Number(event.target.value) : null })} />
                <small aria-hidden="true">lux</small>
              </span>
            </label>}
          </div>;
        })}
      </div>
    </fieldset>
    {cutoffs.length > 0 && <fieldset className="rule-chat-cutoffs">
      <legend>{t('ruleChatCutoffs')}</legend>
      <CutoffTags filters={cutoffs} onRemove={index => onChange({ ...rule, cutoffs: cutoffs.filter((_, i) => i !== index) })} />
    </fieldset>}
  </>;
}

export function RulePopover({ id, anchor, trigger, result, rule, busy, onChange, onDismiss, onCancel, onApply }: {
  id: string;
  anchor: RefObject<HTMLSpanElement | null>;
  trigger: RefObject<HTMLButtonElement | null>;
  result: ResultMessage;
  rule: Rule;
  busy: boolean;
  onChange: (rule: Rule) => void;
  onDismiss: () => void;
  onCancel: () => void;
  onApply: () => void;
}) {
  const { t, locale } = useI18n();
  const dialog = useRef<HTMLDialogElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const closing = useRef(false);
  const title = useId();
  const valid = validRule(rule);
  const cutoffs = rule.cutoffs ?? [];
  const rows = valid ? rank(result.data.rows, { preference: 'custom', rule }) : [];
  const qualified = rows.filter(row => row.eligible).length;
  const closest = valid && cutoffs.length && !qualified ? rankAngles(result.data.rows, cutoffs).closest : null;
  const angles = (ranked: RankedRow[]) => ranked.filter(row => row.recommended)
    .map(row => row.PVSD_Angle_deg + '°').join(locale === 'zh' ? '、' : ', ') || t('none');
  const previous = valid ? angles(rank(result.data.rows, result.selection)) : '';
  const next = angles(rows);

  function finish(action: () => void) {
    closing.current = true;
    action();
    dialog.current?.close();
    trigger.current?.focus({ preventScroll: true });
  }

  useLayoutEffect(() => {
    const panel = dialog.current!;
    const button = anchor.current!;
    const scroller = button.closest('.conversation-scroll');
    const viewport = window.visualViewport;
    const narrow = window.matchMedia('(max-width: 700px)');
    let needsFocus = true;
    let updatingMode = false;

    function position() {
      const left = viewport?.offsetLeft ?? 0;
      const top = viewport?.offsetTop ?? 0;
      const width = viewport?.width ?? window.innerWidth;
      const height = viewport?.height ?? window.innerHeight;
      const edge = 8;
      if (narrow.matches) {
        panel.style.left = left + 'px';
        panel.style.top = '';
        panel.style.bottom = Math.max(0, window.innerHeight - top - height) + 'px';
        panel.style.maxWidth = width + 'px';
        panel.style.maxHeight = Math.min(height - edge, height * .88) + 'px';
      } else {
        const bounds = scroller?.getBoundingClientRect();
        const frame = button.getBoundingClientRect();
        if (bounds && (frame.bottom <= bounds.top || frame.top >= bounds.bottom)) {
          panel.style.visibility = 'hidden';
          return;
        }
        const upper = Math.max(top, bounds?.top ?? top) + edge;
        const lower = Math.min(top + height, bounds?.bottom ?? top + height) - edge;
        const start = Math.max(upper, frame.bottom + 6);
        const end = Math.min(lower, frame.top - 6);
        const below = lower - start;
        const above = end - upper;
        const placeBelow = below >= 415 || below >= above;
        const room = placeBelow ? below : above;
        // At short viewport heights, give the editor room to scroll even if it overlaps the trigger.
        const overlap = room < 270 && lower - upper > room;
        panel.style.bottom = '';
        panel.style.maxWidth = Math.max(0, width - edge * 2) + 'px';
        panel.style.maxHeight = Math.max(0, Math.min(590, overlap ? lower - upper : room)) + 'px';
        panel.style.left = Math.max(left + edge, Math.min(frame.left, left + width - edge - panel.offsetWidth)) + 'px';
        panel.style.top = (overlap ? Math.max(upper, Math.min(frame.bottom + 6, lower - panel.offsetHeight))
          : placeBelow ? start : end - panel.offsetHeight) + 'px';
      }
      panel.style.visibility = 'visible';
      if (needsFocus) {
        heading.current?.focus({ preventScroll: true });
        needsFocus = false;
      }
    }

    function show() {
      updatingMode = true;
      if (panel.open) panel.close();
      if (narrow.matches) panel.showModal();
      else panel.show();
      needsFocus = true;
      position();
      updatingMode = false;
    }

    function dismiss(event: Event) {
      if (closing.current || updatingMode || narrow.matches || !(event.target instanceof Node)
        || panel.contains(event.target) || button.contains(event.target)) return;
      onDismiss();
    }

    show();
    const observer = new ResizeObserver(position);
    observer.observe(panel);
    observer.observe(button);
    if (scroller) observer.observe(scroller);
    scroller?.addEventListener('scroll', position, { passive: true });
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('focusin', dismiss);
    window.addEventListener('resize', position);
    viewport?.addEventListener('resize', position);
    viewport?.addEventListener('scroll', position);
    narrow.addEventListener('change', show);
    return () => {
      observer.disconnect();
      scroller?.removeEventListener('scroll', position);
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('focusin', dismiss);
      window.removeEventListener('resize', position);
      viewport?.removeEventListener('resize', position);
      viewport?.removeEventListener('scroll', position);
      narrow.removeEventListener('change', show);
    };
  }, [anchor, onDismiss]);

  return createPortal(<dialog id={id} ref={dialog} className="rule-popover" aria-labelledby={title}
    lang={locale === 'zh' ? 'zh-CN' : 'en'}
    onCancel={event => { event.preventDefault(); finish(onDismiss); }}
    onKeyDown={event => {
      if (event.key === 'Escape' && !event.nativeEvent.isComposing) {
        event.preventDefault();
        event.stopPropagation();
        finish(onDismiss);
      }
    }}
    onClick={event => {
      if (event.target !== event.currentTarget) return;
      const rect = event.currentTarget.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) finish(onDismiss);
    }}>
    <header className="rule-popover-heading">
      <h3 id={title} ref={heading} tabIndex={-1}>{t('editPreference')}</h3>
      <button type="button" className="icon-button" aria-label={t('close')} onClick={() => finish(onDismiss)}><X size={17} aria-hidden="true" /></button>
    </header>
    <div className="rule-popover-body">
      <RuleControls rule={rule} onChange={onChange} />
      <RuleDetails rule={rule} rows={rows} hidden={!valid} />
    </div>
    <footer className="rule-popover-footer">
      <div className="rule-preview-summary" role="status" aria-label={t('rulePreview')}>
        {valid ? <>
          <p><strong>{t('rulePreviewRecommendation', { angles: previous === next ? next : previous + ' → ' + next })}</strong>
            {cutoffs.length > 0 && <span className="muted"> · {t('ruleQualifiedCount', { qualified, total: rows.length })}</span>}</p>
          {closest && <><p>{t('rankingNone', { angle: closest.row.PVSD_Angle_deg })}</p><CutoffMisses closest={closest} /></>}
        </> : <p className="warning-text">{t(daylightWeightTotal(rule) === 100 ? 'ruleIncomplete' : 'rulePreviewNeedsWeights')}</p>}
      </div>
      <div className="rule-popover-actions">
        <button type="button" className="button secondary" onClick={() => finish(onCancel)}>{t('cancel')}</button>
        <button type="button" className="button primary" disabled={busy || !valid} onClick={() => finish(onApply)}>{t('applyRule')}</button>
      </div>
    </footer>
  </dialog>, document.body);
}
