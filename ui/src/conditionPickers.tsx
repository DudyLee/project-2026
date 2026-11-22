import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { MonthSelect, RoomOptions } from './controls';
import { useI18n } from './i18n';
import { displayedHour, HOURS, REFERENCE_YEAR, type Conditions, type Room } from './types';

type Field = keyof Conditions;
type Picker = { field: Field; mode: 'hover' | 'explicit' };

const ENTER_DELAY = 150;
const LEAVE_DELAY = 200;
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

export function ConditionPickers({ value, rooms, disabled, unsimulated, onChange }: {
  value: Conditions;
  rooms: Room[];
  disabled: boolean;
  unsimulated: boolean;
  onChange: (change: Partial<Conditions>) => void;
}) {
  const { date, locale, t } = useI18n();
  const id = useId();
  const [open, setOpen] = useState<Picker | null>(null);
  const popup = useRef<HTMLDivElement>(null);
  const triggers = useRef<Partial<Record<Field, HTMLButtonElement | null>>>({});
  const timer = useRef<number | undefined>(undefined);
  const field = open?.field;
  const mode = open?.mode;

  const cancelDelay = useCallback(() => window.clearTimeout(timer.current), []);

  const close = useCallback((restoreFocus = false) => {
    cancelDelay();
    if (restoreFocus && field) triggers.current[field]?.focus({ preventScroll: true });
    setOpen(null);
  }, [cancelDelay, field]);

  useEffect(() => cancelDelay, [cancelDelay]);
  useEffect(() => {
    if (disabled) close();
  }, [disabled, close]);

  useLayoutEffect(() => {
    const panel = popup.current;
    const trigger = field && triggers.current[field];
    if (!panel || !trigger) return;
    const scroller = trigger.closest('.conversation-scroll');
    const viewport = window.visualViewport;

    function position() {
      if (!panel || !trigger) return;
      const anchor = trigger.getBoundingClientRect();
      const bounds = scroller?.getBoundingClientRect();
      if (bounds && (anchor.bottom < bounds.top || anchor.top > bounds.bottom)) {
        close();
        return;
      }
      const viewportTop = viewport?.offsetTop ?? 0;
      const viewportLeft = viewport?.offsetLeft ?? 0;
      const top = Math.max(viewportTop, bounds?.top ?? 0) + EDGE;
      const bottom = Math.min(viewportTop + (viewport?.height ?? window.innerHeight), bounds?.bottom ?? window.innerHeight) - EDGE;
      const left = viewportLeft + EDGE;
      const right = viewportLeft + (viewport?.width ?? window.innerWidth) - EDGE;
      const belowStart = Math.max(top, anchor.bottom + GAP);
      const aboveEnd = Math.min(bottom, anchor.top - GAP);
      const below = bottom - belowStart;
      const above = aboveEnd - top;
      const useBelow = below >= panel.scrollHeight + 2 || below >= above;
      panel.style.maxWidth = Math.max(0, right - left) + 'px';
      panel.style.maxHeight = Math.max(0, useBelow ? below : above) + 'px';
      panel.style.left = Math.max(left, Math.min(anchor.left, right - panel.offsetWidth)) + 'px';
      panel.style.top = (useBelow ? belowStart : aboveEnd - panel.offsetHeight) + 'px';
      panel.style.visibility = 'visible';
    }

    position();
    const observer = new ResizeObserver(position);
    observer.observe(panel);
    observer.observe(trigger);
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
  }, [field, close]);

  useLayoutEffect(() => {
    if (mode === 'explicit' && !popup.current?.contains(document.activeElement)) {
      popup.current?.querySelector<HTMLElement>('[data-picker-focus]')?.focus({ preventScroll: true });
    }
  }, [field, mode]);

  useEffect(() => {
    if (!field) return;
    function dismiss(event: Event) {
      const target = event.target;
      if (!(target instanceof Node) || popup.current?.contains(target)
        || Object.values(triggers.current).some(trigger => trigger?.contains(target))) return;
      close();
    }
    function escape(event: KeyboardEvent) {
      if (event.key === 'Escape') close(Boolean(popup.current?.contains(document.activeElement)));
    }
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('focusin', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('focusin', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [field, close]);

  function leave() {
    cancelDelay();
    if (open?.mode === 'hover') timer.current = window.setTimeout(() => setOpen(null), LEAVE_DELAY);
  }

  function choose(change: Partial<Conditions>) {
    close(true);
    onChange(change);
  }

  const fields = [
    { field: 'room_id', label: value.room_id, name: 'changeRoom', prompt: 'chooseRoom' },
    { field: 'day', label: date(value.day), name: 'changeDate', prompt: 'chooseDate' },
    { field: 'hour', label: displayedHour(value.hour), name: 'changeHour', prompt: 'chooseHour' },
  ] as const;

  return <>
    <h2 className="card-conditions">
      {fields.map(({ field: key, label, name }, index) => (
        <span key={key} className={key === 'room_id' ? 'condition-room' : 'condition-time'}>
          {index === 2 && <span className="condition-divider" aria-hidden="true">·</span>}
          <button ref={node => { triggers.current[key] = node; }} type="button" className="condition-trigger" disabled={disabled}
            aria-label={t(name, { value: label })} aria-haspopup="dialog" aria-expanded={field === key}
            aria-controls={field === key ? id : undefined}
            onPointerEnter={event => {
              if (event.pointerType !== 'mouse') return;
              cancelDelay();
              if (open?.mode !== 'explicit' && field !== key) {
                timer.current = window.setTimeout(() => setOpen({ field: key, mode: 'hover' }), ENTER_DELAY);
              }
            }}
            onPointerLeave={leave}
            onClick={() => {
              cancelDelay();
              if (field === key && mode === 'explicit') close();
              else setOpen({ field: key, mode: 'explicit' });
            }}>{label}</button>
        </span>
      ))}
      {unsimulated && <span className="condition-coverage">{t('noSimulation')}</span>}
    </h2>
    {field && createPortal(
      <div ref={popup} id={id} role="dialog" lang={locale === 'zh' ? 'zh-CN' : 'en'}
        aria-label={t(fields.find(item => item.field === field)!.prompt)}
        className={'condition-popup ' + (field === 'room_id' ? 'room-popup' : '')}
        style={{ '--picker-width': field === 'room_id' ? '320px' : field === 'day' ? '282px' : '250px' } as CSSProperties}
        onPointerEnter={cancelDelay} onPointerLeave={leave}
        onFocusCapture={() => {
          cancelDelay();
          if (mode === 'hover') setOpen({ field, mode: 'explicit' });
        }}>
        {field === 'room_id' && <RoomOptions rooms={rooms} current={value.room_id} showCurrent
          onChange={room_id => choose({ room_id })} />}
        {field === 'day' && <DateOptions current={value.day} choose={day => choose({ day })} />}
        {field === 'hour' && <div className="condition-grid hour-options">
          {HOURS.map(hour => <button type="button" key={hour} aria-pressed={hour === value.hour}
            data-picker-focus={hour === value.hour ? '' : undefined}
            onClick={() => choose({ hour })}>{displayedHour(hour)}</button>)}
        </div>}
      </div>, document.body)}
  </>;
}
