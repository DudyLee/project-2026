import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { observer } from 'mobx-react-lite';
import {
  Send, Building2, CalendarDays, ChartNoAxesColumnIncreasing, CheckCircle2, Clock3, Download, FlaskConical,
  Layers3, MessageCircle, PanelRightClose, PanelRightOpen, Plus, SlidersHorizontal, Sun, X, Zap,
} from 'lucide-react';
import { CandidateTimes, DateFields, HourSelect, PreferenceButtons, RoomPicker, RuleEditor } from './controls';
import { ExportDialog, ResultCard, SideResults, type ExportTarget } from './results';
import { validRule } from './scoring';
import { builtInLocale, isLocale, LocaleProvider, text, useI18n } from './i18n';
import { Workspace } from './store';
import { displayedHour, getConditions, HOURS, type Issue } from './types';
import './style.css';

const store = new Workspace();

function Mark({ small = false }: { small?: boolean }) {
  return <span className={'mark' + (small ? ' small-mark' : '')}><Layers3 size={small ? 16 : 22} strokeWidth={2.1} aria-hidden="true" /></span>;
}

const Navigation = observer(function Navigation() {
  const { date, t } = useI18n();
  return <nav className="navigation" aria-label={t('workspace')}>
    <div className="brand"><Mark /><span><strong>Tower A</strong><small>{t('brand')}</small></span></div>
    <button type="button" className="new-chat" onClick={store.newChat}><Plus size={18} />{t('newChat')}</button>
    <div className="nav-history">
      <h2>{t('history')}</h2>
      {store.results.length ? store.results.map(result => <a href={'#' + result.id} key={result.id}>
        <ChartNoAxesColumnIncreasing size={16} /><span><strong>{result.data.room.id}</strong>
          <small>{date(result.data.day)} · {displayedHour(result.data.hour)}</small></span>
      </a>) : <p>{t('emptyHistory')}</p>}
    </div>
    <div className="nav-bottom"><span>{t('trial')}</span><small>{t('modelShort')}</small></div>
  </nav>;
});

function Welcome() {
  const { t } = useI18n();
  return <section className="welcome">
    <div className="welcome-icon"><Sun size={31} strokeWidth={1.6} aria-hidden="true" /></div>
    <h1>{t('welcome')}</h1>
    <p>{t('welcomeInfo')}</p>
    <div className="starter-grid">
      <button type="button" className="starter" onClick={() => void store.send(t('sampleQuery'))}>
        <span className="starter-icon"><ChartNoAxesColumnIncreasing size={18} /></span>
        <span><strong>{t('sample')}</strong><small>{t('sampleDetail')}</small></span>
      </button>
      <button type="button" className="starter" onClick={() => void store.send(t('floorQuery'))}>
        <span className="starter-icon"><Building2 size={18} /></span>
        <span><strong>{t('floorSample')}</strong><small>{t('floorDetail')}</small></span>
      </button>
    </div>
  </section>;
}

const Clarification = observer(function Clarification() {
  const { t, number } = useI18n();
  function title(issue: Issue): string {
    if (issue.field === 'room') return issue.reason === 'not_found' ? t('roomInvalid') : t('chooseRoom');
    const isDay = issue.field === 'day';
    if (issue.reason === 'out_of_range') return isDay ? t('dateRange') : t('hourRange');
    if (issue.reason === 'multiple') return isDay ? t('dateOptions') : t('hourOptions');
    return isDay ? t('chooseDate') : t('chooseHour');
  }

  return <div className="clarification">
    {store.issues.map(issue => <div className="clarification-field" key={issue.field}>
      <h3>{issue.field === 'room' ? <Building2 size={17} /> : issue.field === 'day' ? <CalendarDays size={17} /> : <Clock3 size={17} />}
        {title(issue)}</h3>
      {issue.field === 'room' && <>
        {issue.rooms.length > 0 && <div className="room-candidates">
          {issue.rooms.map(room => <button type="button" className="room-option" key={room.id}
            onClick={() => void store.setClarification('room_id', room.id)}>
            <strong>{room.id}</strong><small>{room.level} · {number(room.area)} m²</small>
          </button>)}
        </div>}
        <RoomPicker rooms={store.rooms} current={store.draft.room_id} suggested={issue.rooms} floor={issue.floor}
          onChange={room => void store.setClarification('room_id', room)} />
      </>}
      {issue.field === 'day' && <>
        {issue.options.length > 0 && <CandidateTimes values={issue.options} isDay choose={day => void store.setClarification('day', day)} />}
        <DateFields value={store.draft.day} onChange={day => void store.setClarification('day', day)} />
      </>}
      {issue.field === 'hour' && <CandidateTimes values={issue.options.length > 0 ? issue.options : HOURS}
        choose={hour => void store.setClarification('hour', hour)} />}
    </div>)}
  </div>;
});

const Composer = observer(function Composer() {
  const { t } = useI18n();
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void store.send();
  }
  return <div className="composer-area">
    <form className="composer" onSubmit={submit}>
      <textarea rows={2} placeholder={t('inputExample')}
        aria-label={t('input')} maxLength={1000} value={store.composer}
        onChange={event => store.setComposer(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            void store.send();
          }
        }} />
      <button type="submit" className="send" aria-label={t('send')} title={t('send')} disabled={!store.composer.trim() || store.busy}><Send size={19} /></button>
    </form>
    <p>{(store.health?.agents_available === false || store.interpretation === 'local') && t('localParsing') + ' · '}{t('hoursNote')} · {t('simulationWeather')}</p>
  </div>;
});

const Workbench = observer(function Workbench({ openExport }: { openExport: (target: ExportTarget) => void }) {
  const { t, date } = useI18n();
  const prediction = store.sidePrediction;
  const selection = store.sideSelection;
  const isValid = validRule(selection.rule);
  const ready = !!getConditions(store.sideDraft) && store.health?.prediction_ready;
  return <aside className="workbench" aria-label={t('panel')}>
    <header className="panel-head">
      <span className="panel-icon"><FlaskConical size={19} /></span>
      <h2>{t('panel')}</h2>
      <button type="button" className="icon-button" title={t('closePanel')} aria-label={t('closePanel')} onClick={store.togglePanel}><X size={19} /></button>
    </header>
    <div className="panel-content">
      <section className="panel-section">
        <h3>{t('conditions')}</h3>
        <div className="input-group"><span>{t('room')}</span>
          <RoomPicker key={'r' + store.session} rooms={store.rooms} current={store.sideDraft.room_id}
            onChange={value => store.setSideField('room_id', value)} />
        </div>
        <div className="input-group"><span>{t('date')}</span>
          <DateFields key={'d' + store.session} value={store.sideDraft.day} onChange={value => store.setSideField('day', value)} />
        </div>
        <div className="input-group"><span>{t('hour')}</span>
          <HourSelect value={store.sideDraft.hour} onChange={value => store.setSideField('hour', value)} />
        </div>
        <button type="button" className="button primary run-button" disabled={!ready || store.sideBusy}
          onClick={() => void store.predictSide()}><ChartNoAxesColumnIncreasing size={17} />
          {t(store.sideBusy ? 'running' : prediction ? 'rerun' : 'run')}</button>
        {store.sideError && <p className="warning-text small" role="alert">{t(store.sideError)}</p>}
      </section>
      <section className="panel-section">
        <h3><SlidersHorizontal size={16} />{t('recommendationRules')} <small>{t('optional')}</small></h3>
        <PreferenceButtons value={selection.preference} onChange={store.setSidePreference} allowNone />
        {selection.preference === 'custom' && <RuleEditor rule={selection.rule} onChange={store.editSideRule} />}
      </section>
      {prediction && <section className="panel-section panel-result">
        <h3>{t('sideResults')}</h3>
        <p className="small muted result-reference">{prediction.room.id} · {date(prediction.day)} {displayedHour(prediction.hour)}
          {store.sideDirty && ' · ' + t('previous')}</p>
        <SideResults prediction={prediction} selection={selection} />
        <div className="panel-actions">
          <button type="button" className="button secondary" onClick={() => openExport({ prediction, selection })} disabled={!isValid}><Download size={15} />{t('export')}</button>
          <button type="button" className="button primary" onClick={store.moveSideToChat}
            disabled={store.sideDirty || store.sideBusy || store.busy || !isValid}><MessageCircle size={15} />{t('move')}</button>
        </div>
      </section>}
    </div>
  </aside>;
});

const AppShell = observer(function AppShell() {
  const { locale, t } = useI18n();
  const [exportTarget, openExport] = useState<ExportTarget | null>(null);
  const tail = useRef<HTMLDivElement>(null);
  const lastClarification = store.messages.filter(message => message.kind === 'clarify').at(-1)?.id;
  const messageCount = store.messages.length;

  useEffect(() => { void store.bootstrap(); }, []);
  useEffect(() => {
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    document.title = t('title');
  }, [locale, t]);
  useEffect(() => {
    if (messageCount) tail.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [messageCount]);

  function botMessage(id: string, children: ReactNode, error = false, language?: string) {
    return <div className={'bot-message' + (error ? ' error-message' : '')} key={id} lang={language}>
      <Mark small /><div className="bot-content">{children}</div>
    </div>;
  }

  return <div className="app-shell">
    <Navigation />
    <main className="main-area">
      <header className="topbar">
        <div className="topbar-left"><MessageCircle size={18} /><span>{t('chat')}</span></div>
        <div className="topbar-right">
          <button type="button" className="icon-button mobile-new" title={t('newChat')} aria-label={t('newChat')} onClick={store.newChat}><Plus size={19} /></button>
          {store.health?.prediction_ready ? <span className="service-status"><CheckCircle2 size={15} />{t('ready')}</span>
            : !store.connected || store.health ? <span className="service-status warning-text">{t('notReady')}</span> : null}
          <button type="button" className="button panel-toggle" onClick={store.togglePanel} aria-expanded={store.panel} aria-label={t('panel')}>
            {store.panel ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}
            <span>{t('panel')}</span>
          </button>
          <select className="field language-select" aria-label={t('language')} value={store.uiLocale}
            onChange={event => { if (isLocale(event.target.value)) store.setLocale(event.target.value); }}>
            <option value="en" lang="en">English</option>
            <option value="zh" lang="zh-CN">简体中文</option>
          </select>
        </div>
      </header>
      <div className="conversation-scroll">
        <div className="conversation">
          {!messageCount && <Welcome />}
          {store.messages.map(message => {
            if (message.kind === 'user') return <div key={message.id} className="user-message"><p>{message.text}</p></div>;
            if (message.kind === 'result') return <LocaleProvider key={message.id} locale={builtInLocale(message.language, locale)}>
              <ResultCard result={message} busy={store.busy}
                onPreference={preference => store.choosePreference(message, preference)}
                onOpenSide={() => store.openSide(message)} onExplain={() => store.askAbout(message)}
                onExport={() => openExport({ prediction: message.data, selection: message.selection })} />
            </LocaleProvider>;
            if (message.kind === 'clarify') {
              const active = lastClarification === message.id && store.issues.length > 0;
              const activeLocale = store.conversationLocale;
              const translated = message.language === store.replyLanguage && !isLocale(message.language);
              return botMessage(message.id, <>
                <p>{active ? (translated ? message.text : text(activeLocale, 'clarify')) : t('clarified')}</p>
                {active && <LocaleProvider locale={activeLocale}><Clarification /></LocaleProvider>}
              </>, false, active ? (translated ? message.language : activeLocale) : locale);
            }
            return botMessage(message.id, <p>{message.text}</p>, message.kind === 'error', message.language);
          })}
          {store.busy && botMessage('waiting', <LocaleProvider locale={store.conversationLocale}><Waiting /></LocaleProvider>)}
          {!store.connected && <div className="service-notice" role="status">{t('noConnection')}<button type="button" className="text-button" onClick={() => void store.bootstrap()}>{t('reconnect')}</button></div>}
          <div ref={tail} />
        </div>
      </div>
      <Composer />
    </main>
    {store.panel && <Workbench openExport={openExport} />}
    {exportTarget && <ExportDialog target={exportTarget} onClose={() => openExport(null)} />}
  </div>;
});

function Waiting() {
  const { t } = useI18n();
  return <p className="waiting" role="status"><span />{t('processing')}</p>;
}

const App = observer(function App() {
  return <LocaleProvider locale={store.uiLocale}><AppShell /></LocaleProvider>;
});

createRoot(document.getElementById('root')!).render(<App />);
