import { useEffect, useId, useLayoutEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { observer } from 'mobx-react-lite';
import {
  Send, Building2, CalendarDays, ChartNoAxesColumnIncreasing, Check, CheckCircle2, Clock3,
  Layers3, Pencil, Plus, SlidersHorizontal, Sun, X,
} from 'lucide-react';
import { CandidateTimes, DateFields, HourSelect, RoomPicker } from './controls';
import { ComparisonCard, ExportDialog, RankingCard, ResultCard } from './results';
import { PredictionRail } from './predictionNavigation';
import { builtInLocale, isLocale, LocaleProvider, text, useI18n } from './i18n';
import { Workspace } from './store';
import { ANGLES, displayedHour, getConditions, HOURS, type Issue } from './types';
import './style.css';

const store = new Workspace();

function Mark({ small = false }: { small?: boolean }) {
  return <span className={'mark' + (small ? ' small-mark' : '')}><Layers3 size={small ? 16 : 22} strokeWidth={2.1} aria-hidden="true" /></span>;
}

const ChatTitle = observer(function ChatTitle() {
  const { t } = useI18n();
  const [draft, setDraft] = useState<string | null>(null);
  const editing = draft !== null;
  const input = useRef<HTMLInputElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const restoreFocus = useRef(false);

  useLayoutEffect(() => {
    if (editing) {
      input.current?.focus();
      input.current?.select();
    } else if (restoreFocus.current) {
      trigger.current?.focus();
      restoreFocus.current = false;
    }
  }, [editing]);

  function finish(save: boolean) {
    if (save && draft !== null) store.renameChat(draft);
    restoreFocus.current = true;
    setDraft(null);
  }

  return <div className="topbar-title">
    {editing ? <form className="chat-title-editor" onSubmit={event => { event.preventDefault(); finish(true); }}>
      <input ref={input} className="field" type="text" maxLength={100} aria-label={t('chatName')}
        value={draft} onChange={event => setDraft(event.target.value)}
        onKeyDown={event => {
          if (event.nativeEvent.isComposing) return;
          if (event.key === 'Escape') { event.preventDefault(); finish(false); }
        }} />
      <button type="submit" className="icon-button" aria-label={t('saveChatName')} disabled={!draft.trim()}>
        <Check size={16} aria-hidden="true" />
      </button>
      <button type="button" className="icon-button" aria-label={t('cancel')} onClick={() => finish(false)}>
        <X size={16} aria-hidden="true" />
      </button>
    </form> : <button type="button" className="chat-title" ref={trigger}
      aria-label={t('editChatName', { name: store.chatTitle })} title={store.chatTitle}
      onClick={() => setDraft(store.chatTitle)}>
      <span>{store.chatTitle}</span><Pencil size={13} aria-hidden="true" />
    </button>}
  </div>;
});

const ModelStatus = observer(function ModelStatus() {
  const { t } = useI18n();
  const id = useId();
  const status = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<'hover' | 'focus' | 'tap' | null>(null);
  const health = store.health;

  useEffect(() => {
    if (!mode) return;
    function dismiss(event: PointerEvent) {
      if (event.target instanceof Node && !status.current?.contains(event.target)) setMode(null);
    }
    function escape(event: KeyboardEvent) {
      if (event.key === 'Escape') setMode(null);
    }
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [mode]);

  if (!health?.prediction_ready) return !store.connected || health
    ? <span className="service-status warning-text">{t('notReady')}</span> : null;

  return <div className="model-status" ref={status} onPointerLeave={() => {
    if (mode === 'hover') setMode(null);
  }}>
    <button type="button" className="service-status model-status-trigger" aria-label={t('ready')}
      aria-expanded={Boolean(mode)} aria-describedby={mode ? id : undefined}
      onPointerEnter={event => { if (event.pointerType === 'mouse' && !mode) setMode('hover'); }}
      onFocus={event => { if (event.currentTarget.matches(':focus-visible')) setMode('focus'); }}
      onBlur={() => setMode(null)} onClick={() => setMode(mode === 'tap' ? null : 'tap')}>
      <CheckCircle2 size={15} aria-hidden="true" /><span>{t('ready')}</span>
    </button>
    {mode && <div className="model-tooltip" id={id} role="tooltip">
      {health.model_version && <strong>{t('modelVersion', { version: health.model_version })}</strong>}
      {health.weather_source === 'original_simulation_weather' && <p>{t('simulationWeather')}</p>}
      <p>{t('hoursNote')}</p>
    </div>}
  </div>;
});

const Welcome = observer(function Welcome() {
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
    {store.conditionDraft ? <ConditionsEditor /> :
      <button type="button" className="text-button manual-start" onClick={() => store.openConditions()}><SlidersHorizontal size={16} />{t('manualConditions')}</button>}
  </section>;
});

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
    <p>{(store.health?.agents_available === false || store.interpretation === 'local') && t('localParsing') + ' · '}{t('hoursNote')}</p>
  </div>;
});

const ConditionsEditor = observer(function ConditionsEditor() {
  const { t } = useI18n();
  const draft = store.conditionDraft;
  if (!draft) return null;
  return <section className="inline-editor" aria-label={t('conditions')}>
    <h3>{t('conditions')}</h3>
    <div className="editor-fields">
      <div className="input-group"><span>{t('room')}</span>
        <RoomPicker rooms={store.rooms} current={draft.room_id}
          onChange={value => store.setCondition('room_id', value)} />
      </div>
      <div className="input-group"><span>{t('date')}</span>
        <DateFields value={draft.day} onChange={value => store.setCondition('day', value)} />
      </div>
      <div className="input-group"><span>{t('hour')}</span>
        <HourSelect value={draft.hour} onChange={value => store.setCondition('hour', value)} />
      </div>
    </div>
    <div className="editor-actions">
      <button type="button" className="button secondary" onClick={store.closeConditions}>{t('cancel')}</button>
      <button type="button" className="button primary" disabled={store.busy || !getConditions(draft) || !store.health?.prediction_ready}
        onClick={store.runConditions}>{t(store.busy ? 'running' : 'run')}</button>
    </div>
  </section>;
});

const ContextChoices = observer(function ContextChoices({ kind }: { kind: 'choose-result' | 'choose-angles' }) {
  const { t, date } = useI18n();
  const angles = store.angleChoice;
  if (kind === 'choose-angles' && angles) return <div className="context-choices">
    {ANGLES.map(angle => <button type="button" className="button secondary" key={angle}
      aria-pressed={angles.angles.includes(angle)} onClick={() => store.toggleCompareAngle(angle)}>{angle}°</button>)}
    <button type="button" className="button primary" disabled={angles.angles.length < 2}
      onClick={store.finishAngleChoice}>{t('compare')}</button>
  </div>;
  if (kind !== 'choose-result' || !store.resultChoice) return null;
  return <div className="context-choices">
    {store.resultChoice.ids.map(id => {
      const result = store.findResult(id);
      if (!result) return null;
      return <button type="button" className="room-option" key={id} disabled={store.busy}
        onClick={() => void store.chooseResult(id)}><strong>{result.data.room.id}</strong>
        <small>{date(result.data.day)} · {displayedHour(result.data.hour)} · {t(result.selection.preference ?? 'noPreference')}</small>
      </button>;
    })}
  </div>;
});

const AppShell = observer(function AppShell() {
  const { locale, t } = useI18n();
  const tail = useRef<HTMLDivElement>(null);
  const conversation = useRef<HTMLDivElement>(null);
  const hasResults = store.results.length > 0;
  const lastClarification = store.messages.filter(message => message.kind === 'clarify').at(-1)?.id;
  const messageCount = store.messages.length;
  const lastChoices = store.messages.filter(message => message.kind === 'choose-result' || message.kind === 'choose-angles').at(-1)?.id;
  const reveal = store.reveal;
  const scrollState = useRef({ messageCount: 0, reveal });

  useEffect(() => { void store.bootstrap(); }, []);
  useEffect(() => {
    document.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
    document.title = t('title');
  }, [locale, t]);
  useEffect(() => {
    const previous = scrollState.current;
    scrollState.current = { messageCount, reveal };
    if (reveal && reveal !== previous.reveal) {
      const card = document.getElementById(reveal.resultId);
      const selector = { editor: '.custom-preference', charts: '.charts', header: '.card-top' }[reveal.target];
      const anchor = card?.querySelector(selector) ?? card;
      anchor?.scrollIntoView({ block: reveal.target === 'header' ? 'start' : 'nearest',
        behavior: reveal.target === 'editor' ? 'instant' : 'smooth' });
    } else if (messageCount && messageCount !== previous.messageCount) {
      tail.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [messageCount, reveal]);

  function botMessage(id: string, children: ReactNode, error = false, language?: string) {
    return <div className={'bot-message' + (error ? ' error-message' : '')} key={id} lang={language}>
      <Mark small /><div className="bot-content">{children}</div>
    </div>;
  }

  return <div className="app-shell">
    <main className={'main-area' + (hasResults ? ' has-predictions' : '')}>
      <header className="topbar">
        <div className="topbar-left">
          <div className="brand"><Mark /><span><strong>Tower A</strong><small>{t('brand')}</small></span></div>
          <button type="button" className="button secondary new-chat" title={t('newChat')} aria-label={t('newChat')} onClick={store.newChat}>
            <Plus size={17} aria-hidden="true" /><span>{t('newChat')}</span>
          </button>
        </div>
        <ChatTitle key={store.chatId} />
        <div className="topbar-right">
          <ModelStatus />
          <select className="field language-select" aria-label={t('language')} value={store.uiLocale}
            onChange={event => { if (isLocale(event.target.value)) store.setLocale(event.target.value); }}>
            <option value="en" lang="en">English</option>
            <option value="zh" lang="zh-CN">简体中文</option>
          </select>
        </div>
      </header>
      <div className="conversation-pane">
        <div className="conversation-scroll" ref={conversation}>
          <div className="conversation">
            {!messageCount && <Welcome />}
            {store.messages.map(message => {
              if (message.kind === 'user') return <div key={message.id} className="user-message"><p>{message.text}</p></div>;
              if (message.kind === 'result') return <LocaleProvider key={message.id} locale={builtInLocale(message.language, locale)}>
                <ResultCard result={message} busy={store.busy} rooms={store.rooms}
                  ruleDraft={store.ruleEditorId === message.id ? store.ruleDrafts.get(message.id) : undefined}
                  onPreference={preference => store.choosePreference(message, preference)}
                  onChangeConditions={change => store.changeConditions(message, change)}
                  onEditPreference={() => store.openRule(message)}
                  onRuleChange={store.setRule} onDismissRule={store.dismissRule}
                  onCancelRule={store.cancelRule} onApplyRule={store.applyRule}
                  onInspect={angle => store.inspect(message, angle)} onLegend={metric => store.toggleMetric(message, metric)}
                  onValuesOpen={open => store.setValuesOpen(message, open)}
                  onExplain={() => store.askAbout(message)} onBookmark={() => store.toggleBookmark(message)}
                  onExport={() => store.openExport(message)} />
              </LocaleProvider>;
              if (message.kind === 'comparison') return <LocaleProvider key={message.id} locale={builtInLocale(message.language, locale)}>
                <ComparisonCard message={message} />
              </LocaleProvider>;
              if (message.kind === 'ranking') return <LocaleProvider key={message.id} locale={builtInLocale(message.language, locale)}>
                <RankingCard message={message} />
              </LocaleProvider>;
              if (message.kind === 'choose-result' || message.kind === 'choose-angles') {
                return botMessage(message.id, <><p>{message.text}</p>{message.id === lastChoices &&
                  <LocaleProvider locale={store.conversationLocale}><ContextChoices kind={message.kind} /></LocaleProvider>}</>,
                  false, message.language);
              }
              if (message.kind === 'clarify') {
                const active = lastClarification === message.id && store.issues.length > 0;
                const activeLocale = store.conversationLocale;
                const translated = message.language === store.replyLanguage && !isLocale(message.language);
                const originalLocale = builtInLocale(message.language, locale);
                return botMessage(message.id, <>
                  <p>{active ? (translated ? message.text : text(activeLocale, 'clarify')) : text(originalLocale, 'clarified')}</p>
                  {active && <LocaleProvider locale={activeLocale}><Clarification /></LocaleProvider>}
                </>, false, active ? (translated ? message.language : activeLocale) : originalLocale);
              }
              return botMessage(message.id, <p>{message.text}</p>, message.kind === 'error', message.language);
            })}
            {messageCount > 0 && !store.results.length && !store.issues.length && !store.busy &&
              (store.conditionDraft ? <ConditionsEditor /> :
                <button type="button" className="text-button manual-start manual-followup" onClick={() => store.openConditions()}>
                  <SlidersHorizontal size={16} />{t('manualConditions')}
                </button>)}
            {store.busy && botMessage('waiting', <LocaleProvider locale={store.conversationLocale}><Waiting /></LocaleProvider>)}
            {!store.connected && <div className="service-notice" role="status">{t('noConnection')}<button type="button" className="text-button" onClick={() => void store.bootstrap()}>{t('reconnect')}</button></div>}
            <div ref={tail} />
          </div>
        </div>
        {hasResults && <PredictionRail store={store} conversation={conversation} />}
      </div>
      <Composer />
    </main>
    {store.exportTarget && <LocaleProvider locale={builtInLocale(store.exportLanguage, locale)}>
      <ExportDialog target={store.exportTarget} onClose={store.closeExport} />
    </LocaleProvider>}
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
