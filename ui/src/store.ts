import { makeAutoObservable, runInAction } from 'mobx';
import { chooseRule, validRule } from './scoring';
import { builtInLocale, DEFAULT_LOCALE, displayedDate, isLocale, text, type CopyKey, type Locale } from './i18n';
import {
  displayedHour, getConditions, type ChatResponse, type Conditions, type Draft, type Issue, type Message,
  type Prediction, type Preference, type ResultMessage, type Room, type Rule, type RuleSelection,
} from './types';

interface Health {
  prediction_ready: boolean;
  agents_available: boolean;
  weather_hours: number;
  room_count: number;
}

const EMPTY: Draft = { room_id: null, day: null, hour: null, preference: null };
const LANGUAGE_STORAGE = 'pvsd-interface-language';
const API_ERRORS = ['modelError', 'roomError', 'weatherError', 'scoreError'] as const;

function storedLocale(): Locale {
  try {
    const locale = window.localStorage.getItem(LANGUAGE_STORAGE);
    return isLocale(locale) ? locale : DEFAULT_LOCALE;
  } catch {
    return DEFAULT_LOCALE;
  }
}

function requestError(error: unknown, fallback: CopyKey): CopyKey {
  return API_ERRORS.find(code => error instanceof Error && error.message === code) ?? fallback;
}

async function api<T>(path: string, payload?: object): Promise<T> {
  const response = await fetch('/api/' + path, payload === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === 'string' ? error.detail : 'requestError');
  }
  return response.json();
}

function conditionsOf(result: Prediction): Conditions {
  return { room_id: result.room.id, day: result.day, hour: result.hour };
}

export class Workspace {
  rooms: Room[] = [];
  health: Health | null = null;
  connected = true;
  uiLocale = storedLocale();
  replyLanguage: string | null = null;
  interpretation: 'agent' | 'local' | null = null;
  messages: Message[] = [];
  draft = { ...EMPTY };
  issues: Issue[] = [];
  busy = false;
  composer = '';
  session = 0;
  panel = typeof window !== 'undefined' && window.innerWidth >= 1100;
  sideDraft = { ...EMPTY };
  sidePrediction: Prediction | null = null;
  sideSelection = chooseRule(null);
  sideBusy = false;
  sideError: CopyKey | null = null;
  private sideEpoch = 0;
  private messageId = 0;
  private focusedResultId: string | null = null;

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });
  }

  async bootstrap() {
    try {
      const [health, data] = await Promise.all([api<Health>('health'), api<{ rooms: Room[] }>('rooms')]);
      runInAction(() => {
        this.health = health;
        this.rooms = data.rooms;
        this.connected = true;
      });
    } catch {
      runInAction(() => { this.connected = false; });
    }
  }

  get results() {
    return this.messages.filter((message): message is ResultMessage => message.kind === 'result');
  }

  get lastResult() {
    return this.results.at(-1);
  }

  get sideDirty() {
    return !this.sidePrediction || this.sideDraft.room_id !== this.sidePrediction.room.id
      || this.sideDraft.day !== this.sidePrediction.day || this.sideDraft.hour !== this.sidePrediction.hour;
  }

  get conversationLanguage() {
    return this.replyLanguage ?? this.uiLocale;
  }

  get conversationLocale() {
    return builtInLocale(this.replyLanguage, this.uiLocale);
  }

  setLocale(locale: Locale) {
    this.uiLocale = locale;
    this.replyLanguage = null;
    try {
      window.localStorage.setItem(LANGUAGE_STORAGE, locale);
    } catch {
      // The selection still applies for this page when storage is unavailable.
    }
  }

  push(message: { kind: 'user' | 'assistant' | 'clarify' | 'error'; text: string; language?: string }
    | { kind: 'result'; data: Prediction; selection: RuleSelection; language?: string }) {
    const id = 'message-' + ++this.messageId;
    this.messages.push({ ...message, id });
    if (message.kind === 'result') this.focusedResultId = id;
  }

  newChat() {
    this.session += 1;
    this.messages = [];
    this.focusedResultId = null;
    this.composer = '';
    this.issues = [];
    this.interpretation = null;
    this.replyLanguage = null;
    this.draft = { ...EMPTY };
    this.busy = false;
    this.sideEpoch += 1;
    this.sideBusy = false;
    this.sideError = null;
    this.sideDraft = { ...EMPTY };
    this.sidePrediction = null;
    this.sideSelection = chooseRule(null);
  }

  setComposer(value: string) {
    this.composer = value;
  }

  togglePanel() {
    this.panel = !this.panel;
  }

  async send(message = this.composer) {
    const query = message.trim();
    if (!query || this.busy) return;
    const session = this.session;
    this.composer = '';
    this.push({ kind: 'user', text: query });
    this.busy = true;
    try {
      const data = await api<ChatResponse>('chat', {
        message: query, draft: this.draft, language: this.conversationLanguage,
      });
      if (session !== this.session) return;
      runInAction(() => {
        if (data.requested_language) this.replyLanguage = data.requested_language;
        this.interpretation = data.interpretation;
        const locale = this.conversationLocale;
        if (data.action === 'language') {
          this.push({
            kind: 'assistant', language: data.localized_reply ? data.language : locale,
            text: data.localized_reply ?? text(locale, isLocale(data.language) ? 'languageAck' : 'languageService'),
          });
          return;
        }
        this.draft = data.draft;
        this.issues = data.issues;
        if (data.action === 'clarify') this.push({
          kind: 'clarify', language: data.localized_reply ? data.language : locale,
          text: data.localized_reply ?? text(locale, 'clarify'),
        });
      });
      if (data.action === 'predict') await this.predictDraft(session, data.localized_reply);
      if (data.action === 'explain') await this.explain(query, session);
    } catch (error) {
      runInAction(() => {
        if (session !== this.session) return;
        this.push({ kind: 'error', language: this.conversationLocale, text: text(this.conversationLocale, requestError(error, 'requestError')) });
        this.panel = true;
      });
    } finally {
      runInAction(() => { if (session === this.session) this.busy = false; });
    }
  }

  async setClarification<K extends 'room_id' | 'day' | 'hour'>(field: K, value: Draft[K]) {
    if (this.busy) return;
    this.draft = { ...this.draft, [field]: value };
    if (value !== null) {
      const issueField = field === 'room_id' ? 'room' : field;
      this.issues = this.issues.filter(issue => issue.field !== issueField);
    }
    if (!getConditions(this.draft) || this.issues.length) return;
    const session = this.session;
    this.busy = true;
    try {
      await this.predictDraft(session);
      if (session === this.session && !isLocale(this.conversationLanguage) && this.health?.agents_available) {
        await this.explain('Summarize the prediction in one sentence.', session);
      }
    } catch (error) {
      runInAction(() => {
        if (session === this.session) this.push({ kind: 'error', language: this.conversationLocale, text: text(this.conversationLocale, requestError(error, 'predictError')) });
      });
    } finally {
      runInAction(() => { if (session === this.session) this.busy = false; });
    }
  }

  private async predictDraft(session: number, introduction: string | null = null) {
    const conditions = getConditions(this.draft);
    if (!conditions) return;
    const last = this.focusedResult;
    const preference = this.draft.preference;
    const selection = preference === 'custom' && last?.selection.preference === 'custom'
      ? last.selection : chooseRule(preference);
    const data = await api<Prediction>('predict', conditions);
    runInAction(() => {
      if (session !== this.session) return;
      if (introduction) this.push({ kind: 'assistant', text: introduction, language: this.conversationLanguage });
      this.push({ kind: 'result', data, selection, language: this.replyLanguage ?? undefined });
      if (!this.sidePrediction && !this.sideBusy && !getConditions(this.sideDraft)) {
        this.setSideFrom(data, selection);
      }
    });
  }

  get focusedResult() {
    return this.results.find(result => result.id === this.focusedResultId) ?? this.lastResult;
  }

  async explain(question: string, session = this.session, result = this.focusedResult) {
    if (!result) return;
    const { selection } = result;
    const detail = await api<{ text: string; language: string }>('explain', {
      conditions: conditionsOf(result.data),
      rule: selection.preference && validRule(selection.rule) ? selection.rule : null,
      question, language: this.conversationLanguage,
    });
    runInAction(() => {
      if (session === this.session) this.push({ kind: 'assistant', text: detail.text, language: detail.language });
    });
  }

  askAbout(result: ResultMessage) {
    if (this.busy) return;
    this.focusedResultId = result.id;
    this.draft = { ...conditionsOf(result.data), preference: result.selection.preference };
    this.issues = [];
    const question = text(this.conversationLocale, result.selection.preference ? 'explainRecommended' : 'explainCompare');
    this.push({ kind: 'user', text: question });
    this.busy = true;
    const session = this.session;
    void this.explain(question, session, result).catch(() => {
      runInAction(() => {
        if (session === this.session) this.push({ kind: 'error', text: text(this.conversationLocale, 'explainError'), language: this.conversationLocale });
      });
    }).finally(() => {
      runInAction(() => { if (session === this.session) this.busy = false; });
    });
  }

  choosePreference(result: ResultMessage, preference: Preference | null) {
    const selection = chooseRule(preference, result.selection.rule);
    if (preference === 'custom') {
      this.setSideFrom(result.data, selection);
      this.panel = true;
      return;
    }
    if (this.focusedResult?.id === result.id && !this.issues.length) this.draft.preference = preference;
    result.selection = selection;
  }

  private setSideFrom(data: Prediction, selection: RuleSelection) {
    this.sideEpoch += 1;
    this.sideBusy = false;
    this.sideError = null;
    this.sidePrediction = data;
    this.sideSelection = { preference: selection.preference, rule: { ...selection.rule } };
    this.sideDraft = { ...conditionsOf(data), preference: selection.preference };
  }

  openSide(result: ResultMessage) {
    this.setSideFrom(result.data, result.selection);
    this.panel = true;
  }

  setSideField<K extends 'room_id' | 'day' | 'hour'>(field: K, value: Draft[K]) {
    this.sideDraft = { ...this.sideDraft, [field]: value };
    this.sideError = null;
    this.sideEpoch += 1;
    this.sideBusy = false;
  }

  setSidePreference(preference: Preference | null) {
    this.sideSelection = chooseRule(preference, this.sideSelection.rule);
    this.sideDraft.preference = preference;
  }

  editSideRule(rule: Rule) {
    this.sideSelection = { preference: 'custom', rule };
    this.sideDraft.preference = 'custom';
  }

  async predictSide() {
    const conditions = getConditions(this.sideDraft);
    if (!conditions || this.sideBusy) return;
    const epoch = ++this.sideEpoch;
    this.sideBusy = true;
    this.sideError = null;
    try {
      const data = await api<Prediction>('predict', conditions);
      runInAction(() => { if (epoch === this.sideEpoch) this.sidePrediction = data; });
    } catch (error) {
      runInAction(() => {
        if (epoch === this.sideEpoch) this.sideError = requestError(error, 'sideError');
      });
    } finally {
      runInAction(() => { if (epoch === this.sideEpoch) this.sideBusy = false; });
    }
  }

  moveSideToChat() {
    const data = this.sidePrediction;
    if (!data || this.sideDirty || this.sideBusy || this.busy || !validRule(this.sideSelection.rule)) return;
    const selection = { preference: this.sideSelection.preference, rule: { ...this.sideSelection.rule } };
    this.draft = { ...conditionsOf(data), preference: selection.preference };
    this.issues = [];
    this.push({
      kind: 'user', text: text(this.uiLocale, 'moved', {
        room: data.room.id, date: displayedDate(data.day, this.uiLocale), hour: displayedHour(data.hour),
      }),
    });
    this.push({ kind: 'result', data, selection, language: this.replyLanguage ?? undefined });
  }
}
