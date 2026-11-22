import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown } from 'lucide-react';
import { MonthSelect, RoomOptions } from './controls';
import { useI18n } from './i18n';
import { displayedHour, HOURS, REFERENCE_YEAR, type Conditions, type Room } from './types';

type Field = keyof Conditions;
type Edit = { origin: Field; field: Field | null; draft: Conditions };

const GAP = 5;
const EDGE = 8;

function DateOptions({ current, choose }: { current: number; choose: (day: number) => void }) {
  const { date, t } = useI18n();
  const selectedMonth = new Date(Date.UTC(REFERENCE_YEAR, 0, current)).getUTCMonth() + 1;
  const [month, setMonth] = useState(selectedMonth);
  const count = new Date(Date.UTC(REFERENCE_YEAR, month, 0)).getUTCDate();
  const first = 1 + (Date.UTC(REFERENCE_YEAR, month - 1, 1) - Date.UTC(REFERENCE_YEAR, 0, 1)) / 86400000;
  const offsets = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 };

  return <>
    <MonthSelect month={month} onChange={setMonth} />
    <div className="condition-grid date-options" role="group" aria-label={t('date')}
      onKeyDown={event => {
        const buttons = Array.from(event.currentTarget.querySelectorAll('button'));
        const offset = offsets[event.key as keyof typeof offsets];
        if (!offset || !(event.target instanceof HTMLButtonElement)) return;
        const next = Math.max(0, Math.min(count - 1, buttons.indexOf(event.target) + offset));
        event.preventDefault();
        buttons[next].focus();
      }}>
      {Array.from({ length: count }, (_, index) => {
        const day = first + index;
        return <button key={day} type="button" aria-label={date(day)} aria-pressed={day === current}
          onClick={() => choose(day)}>{index + 1}</button>;
      })}
    </div>
  </>;
}

export function ConditionPickers({ value, rooms, disabled, unsimulated, onRun }: {
  value: Conditions;
  rooms: Room[];
  disabled: boolean;
  unsimulated: boolean;
  onRun: (conditions: Conditions) => void;
}) {
  const { date, locale, t } = useI18n();
  const id = useId();
  const [edit, setEdit] = useState<Edit | null>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const popup = useRef<HTMLDivElement>(null);
  const triggers = useRef<Partial<Record<Field, HTMLButtonElement | null>>>({});
  const fieldButtons = useRef<Partial<Record<Field, HTMLButtonElement | null>>>({});
  const origin = edit?.origin;
  const field = edit?.field;
  const open = edit !== null;

  const close = useCallback((restoreFocus = false) => {
    if (restoreFocus && origin) triggers.current[origin]?.focus({ preventScroll: true });
    setEdit(null);
  }, [origin]);

  useEffect(() => {
    if (disabled) close();
  }, [disabled, close]);

  useLayoutEffect(() => {
    const panel = popup.current;
    const anchor = heading.current;
    if (!panel || !anchor) return;
    const scroller = anchor.closest('.conversation-scroll');
    const viewport = window.visualViewport;

    function position() {
      if (!panel || !anchor) return;
      const rect = anchor.getBoundingClientRect();
      const bounds = scroller?.getBoundingClientRect();
      if (bounds && (rect.bottom <= bounds.top || rect.top >= bounds.bottom)) {
        close();
        return;
      }
      const viewportTop = viewport?.offsetTop ?? 0;
      const viewportLeft = viewport?.offsetLeft ?? 0;
      const top = Math.max(viewportTop, bounds?.top ?? 0) + EDGE;
      const bottom = Math.min(viewportTop + (viewport?.height ?? window.innerHeight), bounds?.bottom ?? window.innerHeight) - EDGE;
      const left = viewportLeft + EDGE;
      const right = viewportLeft + (viewport?.width ?? window.innerWidth) - EDGE;
      const belowStart = Math.max(top, rect.bottom + GAP);
      const aboveEnd = Math.min(bottom, rect.top - GAP);
      const below = bottom - belowStart;
      const above = aboveEnd - top;
      const useBelow = below >= 360 || below >= above;
      const space = useBelow ? below : above;
      const useFullHeight = space < 230 && bottom - top > space;
      panel.style.maxWidth = Math.max(0, right - left) + 'px';
      panel.style.maxHeight = Math.max(0, useFullHeight ? bottom - top : space) + 'px';
      panel.style.left = Math.max(left, Math.min(rect.left, right - panel.offsetWidth)) + 'px';
      panel.style.top = (useFullHeight ? Math.max(top, Math.min(belowStart, bottom - panel.offsetHeight))
        : useBelow ? belowStart : aboveEnd - panel.offsetHeight) + 'px';
      panel.style.visibility = 'visible';
    }

    position();
    const observer = new ResizeObserver(position);
    observer.observe(panel);
    observer.observe(anchor);
    if (scroller) observer.observe(scroller);
    scroller?.addEventListener('scroll', position, { passive: true });
    window.addEventListener('resize', position);
    viewport?.addEventListener('resize', position);
    viewport?.addEventListener('scroll', position);
    return () => {
      observer.disconnect();
      scroller?.removeEventListener('scroll', position);
      window.removeEventListener('resize', position);
      viewport?.removeEventListener('resize', position);
      viewport?.removeEventListener('scroll', position);
    };
  }, [open, close]);

  useLayoutEffect(() => {
    if (field) popup.current?.querySelector<HTMLElement>('[data-picker-focus]')?.focus({ preventScroll: true });
  }, [field, origin]);

  useEffect(() => {
    if (!open) return;
    function dismiss(event: Event) {
      const target = event.target;
      if (!(target instanceof Node) || popup.current?.contains(target)
        || Object.values(triggers.current).some(trigger => trigger?.contains(target))) return;
      close();
    }
    function escape(event: KeyboardEvent) {
      if (event.key !== 'Escape' || event.defaultPrevented || event.isComposing) return;
      event.preventDefault();
      close(Boolean(popup.current?.contains(document.activeElement)));
    }
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('focusin', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('focusin', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [open, close]);

  function openFrom(key: Field) {
    if (field === key && origin === key) {
      popup.current?.querySelector<HTMLElement>('[data-picker-focus]')?.focus({ preventScroll: true });
      return;
    }
    setEdit(current => ({ origin: key, field: key, draft: current?.draft ?? value }));
  }

  function choose(change: Partial<Conditions>) {
    if (!edit || !field) return;
    fieldButtons.current[field]?.focus({ preventScroll: true });
    setEdit({ ...edit, field: null, draft: { ...edit.draft, ...change } });
  }

  const fields = [
    { key: 'room_id', label: 'room', name: 'changeRoom', display: (conditions: Conditions) => conditions.room_id },
    { key: 'day', label: 'date', name: 'changeDate', display: (conditions: Conditions) => date(conditions.day) },
    { key: 'hour', label: 'hour', name: 'changeHour', display: (conditions: Conditions) => displayedHour(conditions.hour) },
  ] as const;
  const changed = edit && fields.some(({ key }) => edit.draft[key] !== value[key]);

  return <>
    <h2 className="card-conditions" ref={heading}>
      {fields.map(({ key, name, display }, index) => (
        <span key={key} className={key === 'room_id' ? 'condition-room' : 'condition-time'}>
          {index === 2 && <span className="condition-divider" aria-hidden="true">·</span>}
          <button ref={node => { triggers.current[key] = node; }} type="button" className="condition-trigger" disabled={disabled}
            aria-label={t(name, { value: display(value) })} aria-haspopup="dialog" aria-expanded={origin === key}
            aria-controls={open ? id : undefined} onClick={() => openFrom(key)}>{display(value)}</button>
        </span>
      ))}
      {unsimulated && <span className="condition-coverage">{t('noSimulation')}</span>}
    </h2>
    {edit && createPortal(
      <div ref={popup} id={id} role="dialog" lang={locale === 'zh' ? 'zh-CN' : 'en'}
        aria-label={t('conditions')} className="condition-popup">
        <div className="condition-fields">
          {fields.map(({ key, label, display }) => <button key={key} type="button" className="field condition-field"
            ref={node => { fieldButtons.current[key] = node; }} aria-expanded={field === key}
            aria-controls={field === key ? id + '-options' : undefined}
            onClick={() => setEdit({ ...edit, field: field === key ? null : key })}>
            <small>{t(label)}</small><span>{display(edit.draft)}<ChevronDown size={13} aria-hidden="true" /></span>
          </button>)}
        </div>
        {field && <div className="condition-options" id={id + '-options'}>
          {field === 'room_id' && <RoomOptions rooms={rooms} current={edit.draft.room_id} showCurrent
            onChange={room_id => choose({ room_id })} />}
          {field === 'day' && <DateOptions current={edit.draft.day} choose={day => choose({ day })} />}
          {field === 'hour' && <div className="condition-grid hour-options" role="group" aria-label={t('hour')}>
            {HOURS.map(hour => <button type="button" key={hour} aria-pressed={hour === edit.draft.hour}
              data-picker-focus={hour === edit.draft.hour ? '' : undefined}
              onClick={() => choose({ hour })}>{displayedHour(hour)}</button>)}
          </div>}
        </div>}
        <div className="condition-actions">
          <button type="button" className="button secondary" onClick={() => close(true)}>{t('cancel')}</button>
          <button type="button" className="button primary" disabled={disabled || !changed} onClick={() => {
            close(true);
            onRun(edit.draft);
          }}>{t('run')}</button>
        </div>
      </div>, document.body)}
  </>;
}
