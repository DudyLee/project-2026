import { useEffect, useId, useState } from 'react';
import { Building2, Check, ChevronDown, Search } from 'lucide-react';
import { useI18n } from './i18n';
import { displayedHour, HOURS, REFERENCE_YEAR, type Room } from './types';

interface RoomOptionsProps {
  rooms: Room[];
  current: string | null;
  onChange: (value: string) => void;
  suggested?: Room[];
  floor?: number | null;
}

export function RoomPicker({ rooms, current, onChange, suggested, floor }: RoomOptionsProps) {
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
      <RoomOptions rooms={rooms} current={current} suggested={suggested} floor={floor} autoFocus
        onChange={value => { onChange(value); setOpen(false); }} />
    </div>}
  </div>;
}

export function RoomOptions({ rooms, current, onChange, suggested = [], floor = null, autoFocus = false,
  showCurrent = false }: RoomOptionsProps & { autoFocus?: boolean; showCurrent?: boolean }) {
  const { t, number } = useI18n();
  const [query, setQuery] = useState('');
  const normalized = query.trim().toUpperCase();
  const matches = rooms.filter(room => {
    const value = room.id + ' ' + room.level.toUpperCase();
    if (normalized) return value.includes(normalized);
    if (floor === null) return true;
    const level = room.level.toUpperCase().split(' ')[0];
    return floor === 0 ? level === 'GF' || level === 'G' : level === floor + 'F';
  });
  const selected = showCurrent && !normalized ? matches.find(room => room.id === current) : undefined;
  const candidates = suggested.length && !normalized ? suggested
    : selected ? [selected, ...matches.filter(room => room.id !== current).slice(0, 7)] : matches.slice(0, 8);

  return <>
    <label className="search-field">
      <Search size={16} aria-hidden="true" />
      <input autoFocus={autoFocus} data-picker-focus value={query} onChange={event => setQuery(event.target.value)}
        placeholder={t('searchHint')} aria-label={t('searchRoom')} />
    </label>
    <div className="room-options">
      {candidates.map(option => <button key={option.id} type="button" aria-current={current === option.id || undefined}
        onClick={() => onChange(option.id)}>
        <span><strong>{option.id}</strong><small>{option.level} · {number(option.area)} m²</small></span>
        {current === option.id ? <Check size={17} aria-hidden="true" /> : option.simulated && <small>{t('hasRun')}</small>}
      </button>)}
      {!candidates.length && <p className="muted small">{t('noRoom')}</p>}
    </div>
    {matches.length > candidates.length && <p className="muted small room-hint">{t('moreRooms')}</p>}
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
