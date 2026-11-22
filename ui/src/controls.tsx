import { useEffect, useId, useRef, useState } from 'react';
import { Building2, Check, ChevronDown, Search } from 'lucide-react';
import { useI18n } from './i18n';
import { displayedHour, HOURS, REFERENCE_YEAR, type Room } from './types';

interface RoomOptionsProps {
  rooms: Room[];
  current: string | null;
  onChange: (value: string) => void;
  floor?: number | null;
}

const roomIdOrder = new Intl.Collator('en', { numeric: true });

export function RoomPicker({ rooms, current, onChange, floor }: RoomOptionsProps) {
  const { t } = useI18n();
  const id = useId();
  const [open, setOpen] = useState(false);
  const room = rooms.find(room => room.id === current);

  return <div className="room-picker">
    <button type="button" className="field room-trigger" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={id}>
      <Building2 size={17} aria-hidden="true" />
      <span>{room ? room.id + ' · ' + room.level : current || t('chooseRoom')}</span>
      <ChevronDown size={16} aria-hidden="true" />
    </button>
    {open && <div className="room-menu" id={id}>
      <RoomOptions rooms={rooms} current={current} floor={floor} autoFocus
        onChange={value => { onChange(value); setOpen(false); }} />
    </div>}
  </div>;
}

export function RoomOptions({ rooms, current, onChange, floor = null, autoFocus = false,
  showCurrent = false }: RoomOptionsProps & { autoFocus?: boolean; showCurrent?: boolean }) {
  const { t, number } = useI18n();
  const [query, setQuery] = useState('');
  const [focusedRoom, setFocusedRoom] = useState<string | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const normalized = query.trim().toUpperCase();
  const preferred = normalized || (showCurrent ? current : null);
  const matches = rooms.filter(room => {
    const value = room.id + ' ' + room.level.toUpperCase();
    if (normalized) return value.includes(normalized);
    if (floor === null) return true;
    const level = room.level.toUpperCase().split(' ')[0];
    return floor === 0 ? level === 'GF' || level === 'G' : level === floor + 'F';
  }).sort((left, right) => Number(right.id === preferred) - Number(left.id === preferred)
    || roomIdOrder.compare(left.id, right.id));
  const tabRoom = matches.find(room => room.id === focusedRoom)?.id ?? matches[0]?.id;

  return <>
    <label className="search-field">
      <Search size={16} aria-hidden="true" />
      <input autoFocus={autoFocus} data-picker-focus value={query} onChange={event => {
        setQuery(event.target.value);
        setFocusedRoom(null);
        if (list.current) list.current.scrollTop = 0;
      }} onKeyDown={event => {
        if (event.key !== 'ArrowDown' || !matches.length) return;
        event.preventDefault();
        list.current?.querySelector('button')?.focus();
      }}
        placeholder={t('searchHint')} aria-label={t('searchRoom')} />
    </label>
    <div className="room-options" ref={list} onKeyDown={event => {
      if (!['ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key) || !(event.target instanceof HTMLButtonElement)) return;
      const buttons = Array.from(event.currentTarget.querySelectorAll('button'));
      const next = buttons.indexOf(event.target) + (event.key === 'ArrowUp' ? -1 : 1);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
        : Math.max(0, Math.min(buttons.length - 1, next));
      event.preventDefault();
      buttons[index]?.focus();
    }}>
      {matches.map(option => <button key={option.id} type="button" aria-current={current === option.id || undefined}
        onFocus={() => setFocusedRoom(option.id)} tabIndex={option.id === tabRoom ? 0 : -1}
        onClick={() => onChange(option.id)}>
        <span><strong>{option.id}</strong><small>{option.level} · {number(option.area)} m²</small></span>
        {current === option.id ? <Check size={17} aria-hidden="true" /> : option.simulated && <small>{t('hasRun')}</small>}
      </button>)}
      {!matches.length && <p className="muted small">{t('noRoom')}</p>}
    </div>
    {matches.length > 0 && <p className="muted small room-hint" aria-live="polite">
      {t(matches.length === rooms.length ? 'roomCount' : 'roomMatchCount', { count: number(matches.length, 0), total: number(rooms.length, 0) })}
    </p>}
  </>;
}

export function MonthSelect({ month, onChange, allowEmpty = false }: {
  month: number; onChange: (month: number) => void; allowEmpty?: boolean;
}) {
  const { locale, t } = useI18n();
  const format = new Intl.DateTimeFormat(locale === 'zh' ? 'zh-CN' : 'en-US', { month: 'long', timeZone: 'UTC' });
  return <select className="field" value={month || ''} onChange={event => onChange(Number(event.target.value))}
    aria-label={t('month')} data-picker-focus>
    {allowEmpty && <option value="">{t('month')}</option>}
    {Array.from({ length: 12 }, (_, index) => index + 1).map(value =>
      <option key={value} value={value}>{format.format(new Date(Date.UTC(REFERENCE_YEAR, value - 1, 1)))}</option>)}
  </select>;
}

export function DateFields({ value, onChange }: {
  value: number | null;
  onChange: (value: number | null) => void;
}) {
  const { locale, t } = useI18n();
  const date = value === null ? null : new Date(Date.UTC(REFERENCE_YEAR, 0, value));
  const [month, setMonth] = useState(date ? date.getUTCMonth() + 1 : 0);
  const [day, setDay] = useState(date ? date.getUTCDate() : 0);
  useEffect(() => {
    if (value === null) return;
    const selected = new Date(Date.UTC(REFERENCE_YEAR, 0, value));
    setMonth(selected.getUTCMonth() + 1);
    setDay(selected.getUTCDate());
  }, [value]);

  const count = month ? new Date(Date.UTC(REFERENCE_YEAR, month, 0)).getUTCDate() : 0;
  function update(month: number, day: number) {
    setMonth(month);
    setDay(day);
    if (!month || !day) {
      onChange(null);
    } else {
      onChange(1 + (Date.UTC(REFERENCE_YEAR, month - 1, day) - Date.UTC(REFERENCE_YEAR, 0, 1)) / 86400000);
    }
  }

  return <div className="date-fields">
    <MonthSelect month={month} onChange={value => update(value, 0)} allowEmpty />
    <select className="field" value={day || ''} onChange={event => update(month, Number(event.target.value))} disabled={!month} aria-label={t('date')}>
      <option value="">{t('date')}</option>
      {Array.from({ length: count }, (_, index) => index + 1).map(day => <option key={day} value={day}>{locale === 'zh' ? day + ' 日' : day}</option>)}
    </select>
  </div>;
}

export function CandidateTimes({ values, isDay, choose }: {
  values: number[];
  isDay?: boolean;
  choose: (value: number) => void;
}) {
  const { date } = useI18n();
  return <div className={isDay ? 'pills' : 'hour-grid'}>
    {values.map(value => <button key={value} type="button" className="pill" onClick={() => choose(value)}>
      {isDay ? date(value) : displayedHour(value)}
    </button>)}
  </div>;
}

export function HourSelect({ value, onChange }: {
  value: number | null;
  onChange: (value: number | null) => void;
}) {
  const { t } = useI18n();
  return <select className="field" aria-label={t('hour')} value={value ?? ''} onChange={event => onChange(event.target.value ? Number(event.target.value) : null)}>
    <option value="">{t('chooseHour')}</option>
    {HOURS.map(hour => <option key={hour} value={hour}>{displayedHour(hour)}</option>)}
  </select>;
}
