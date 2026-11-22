import { useEffect, useId, useState } from 'react';
import { Building2, Check, ChevronDown, Search } from 'lucide-react';
import { PREFERENCES, validRule } from './scoring';
import { useI18n } from './i18n';
import { displayedHour, HOURS, REFERENCE_YEAR, type Preference, type Room, type Rule } from './types';

export function RoomPicker({ rooms, current, onChange, suggested = [], floor = null }: {
  rooms: Room[];
  current: string | null;
  onChange: (value: string) => void;
  suggested?: Room[];
  floor?: number | null;
}) {
  const { t, number } = useI18n();
  const id = useId();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const room = rooms.find(room => room.id === current);
  const normalized = query.trim().toUpperCase();
  const matches = rooms.filter(room => {
    const value = room.id + ' ' + room.level.toUpperCase();
    if (normalized) return value.includes(normalized);
    if (floor === null) return true;
    const level = room.level.toUpperCase().split(' ')[0];
    return floor === 0 ? level === 'GF' || level === 'G' : level === floor + 'F';
  });
  const candidates = suggested.length && !normalized ? suggested : matches.slice(0, 8);

  function choose(id: string) {
    onChange(id);
    setOpen(false);
    setQuery('');
  }

  return (
    <div className="room-picker">
      <button type="button" className="field room-trigger" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={id}>
        <Building2 size={17} aria-hidden="true" />
        <span>{room ? room.id + ' · ' + room.level : current || t('chooseRoom')}</span>
        <ChevronDown size={16} aria-hidden="true" />
      </button>
      {open && <div className="room-menu" id={id}>
        <label className="search-field">
          <Search size={16} aria-hidden="true" />
          <input autoFocus value={query} onChange={event => setQuery(event.target.value)} placeholder={t('searchHint')} aria-label={t('searchRoom')} />
        </label>
        <div className="room-options">
          {candidates.map(option => <button key={option.id} type="button" onClick={() => choose(option.id)}>
            <span><strong>{option.id}</strong><small>{option.level} · {number(option.area)} m²</small></span>
            {current === option.id ? <Check size={17} aria-hidden="true" /> : option.simulated && <small>{t('hasRun')}</small>}
          </button>)}
          {!candidates.length && <p className="muted small">{t('noRoom')}</p>}
        </div>
        {matches.length > candidates.length && <p className="muted small room-hint">{t('moreRooms')}</p>}
      </div>}
    </div>
  );
}

export function DateFields({ value, onChange }: {
  value: number | null;
  onChange: (value: number | null) => void;
}) {
  const { locale, t } = useI18n();
  const monthFormat = new Intl.DateTimeFormat(locale === 'zh' ? 'zh-CN' : 'en-US', { month: 'long', timeZone: 'UTC' });
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
    <select className="field" value={month || ''} onChange={event => update(Number(event.target.value), 0)} aria-label={t('month')}>
      <option value="">{t('month')}</option>
      {Array.from({ length: 12 }, (_, index) => index + 1).map(month => <option key={month} value={month}>{monthFormat.format(new Date(Date.UTC(REFERENCE_YEAR, month - 1)))}</option>)}
    </select>
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

export function PreferenceButtons({ value, onChange, allowNone = false }: {
  value: Preference | null;
  onChange: (value: Preference | null) => void;
  allowNone?: boolean;
}) {
  const { t } = useI18n();
  return <div className="pills" role="group" aria-label={t('recommendationPreference')}>
    {PREFERENCES.map(preference => <button
      key={preference} type="button" className="pill" aria-pressed={value === preference}
      onClick={() => onChange(allowNone && value === preference ? null : preference)}
    >{t(preference)}</button>)}
  </div>;
}

export function RuleEditor({ rule, onChange }: { rule: Rule; onChange: (rule: Rule) => void }) {
  const { t } = useI18n();
  const sum = rule.udi + rule.f300 + rule.p10;
  return <div className="rule-editor">
    <div className="weights-line"><span>{t('daylightMetric')} <strong>{100 - rule.power}%</strong></span><span>{t('solarEnergy')} <strong>{rule.power}%</strong></span></div>
    <input type="range" min={0} max={100} value={rule.power} aria-label={t('powerWeight')}
      onChange={event => onChange({ ...rule, power: Number(event.target.value) })} />
    <div className="weight-grid">
      {([
        ['udi', 'udiFull', 0, 100],
        ['f300', 'f300', 0, 100],
        ['p10', 'p10', 0, 100],
        ['p10_target', 'p10Target', 1, 10000],
      ] as const).map(([key, label, min, max]) => <label key={key}>
        <span>{t(label)}</span>
        <span className="weight-input">
          <input className="field" type="number" min={min} max={max} value={rule[key]}
            onChange={event => onChange({ ...rule, [key]: Number(event.target.value) })} />
          <small>{key === 'p10_target' ? 'lux' : '%'}</small>
        </span>
      </label>)}
    </div>
    <div className={validRule(rule) ? 'small muted' : 'small warning-text'} aria-live="polite">
      {t('totalWeights', { percent: sum })}{!validRule(rule) && ' ' + t('weightError')}
    </div>
  </div>;
}
