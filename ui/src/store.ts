import { makeAutoObservable, reaction, runInAction } from 'mobx';
import { chooseRule, editRule, validRule } from './scoring';
import { builtInLocale, DEFAULT_LOCALE, displayedDate, isLocale, text, type CopyKey, type Locale } from './i18n';
import {
  ANGLES, displayedHour, getConditions, METRIC_IDS, type Angle, type ChatResponse, type ComparisonMessage, type Conditions,
  type Draft, type ExportTarget, type Issue, type Message, type MetricId, type Prediction, type Preference, type RankingMessage,
  type ResultMessage, type Room, type Rule, type RuleEdit, type RuleSelection,
} from './types';

interface Health {
  prediction_ready: boolean;
  model_version: string | null;
  weather_source: string;
  agents_available: boolean;
  weather_hours: number;
  room_count: number;
}

const EMPTY: Draft = { room_id: null, day: null, hour: null, preference: null };
const LANGUAGE_STORAGE = 'pvsd-interface-language';
const CHAT_STORAGE = 'pvsd-chat-v1:';
const ACTIVE_CHAT_STORAGE = 'pvsd-active-chat-v1';
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

type ResultChoice = { ids: string[]; command: ChatResponse; question: string; role: 'base' | 'against' };
type AngleChoice = { resultId: string; angles: Angle[]; metrics?: MetricId[] };
type ChatSnapshot = {
  messages: Message[];
  title: string;
  composer: string;
  draft: Draft;
  issues: Issue[];
  interpretation: 'agent' | 'local' | null;
  replyLanguage: string | null;
  focusedResultId: string | null;
  pendingPrediction: ChatResponse | null;
  resultChoice: ResultChoice | null;
  angleChoice: AngleChoice | null;
  ruleDrafts: [string, Rule][];
};
type SavedChat = { updatedAt: number; snapshot: string };

export class ChatSession {
  replyLanguage: string | null = null;
  interpretation: 'agent' | 'local' | null = null;
  messages: Message[] = [];
  draft = { ...EMPTY };
  issues: Issue[] = [];
  busy = false;
  composer = '';
  private customChatTitle = '';
  conditionDraft: Draft | null = null;
  ruleEditorId: string | null = null;
  ruleDrafts = new Map<string, Rule>();
  exportTarget: ExportTarget | null = null;
  exportLanguage: string | null = null;
  exportFilter: 'bookmarked' | 'all' = 'bookmarked';
  selectedResultIds = new Set<string>();
  resultChoice: ResultChoice | null = null;
  angleChoice: AngleChoice | null = null;
  reveal: { resultId: string; target: 'charts' | 'editor' | 'header' } | null = null;
  focusedResultId: string | null = null;
  updatedAt: number;
  private pendingPrediction: ChatResponse | null = null;
  private messageId = 0;

  constructor(private workspace: Workspace, readonly id: string, updatedAt = Date.now(), saved?: ChatSnapshot) {
    this.updatedAt = updatedAt;
    if (saved) {
      this.messages = saved.messages;
      this.messageId = saved.messages.length;
      this.customChatTitle = saved.title;
      this.composer = saved.composer;
      this.draft = saved.draft;
      this.issues = saved.issues;
      this.interpretation = saved.interpretation;
      this.replyLanguage = saved.replyLanguage;
      this.focusedResultId = saved.focusedResultId;
      this.pendingPrediction = saved.pendingPrediction;
      this.resultChoice = saved.resultChoice;
      this.angleChoice = saved.angleChoice;
      this.ruleDrafts = new Map(saved.ruleDrafts);
    }
    makeAutoObservable<this, 'workspace'>(this, { workspace: false, id: false }, { autoBind: true });
  }

  get snapshot(): ChatSnapshot {
    return {
      messages: this.messages, title: this.customChatTitle, composer: this.composer,
      draft: this.draft, issues: this.issues, interpretation: this.interpretation, replyLanguage: this.replyLanguage,
      focusedResultId: this.focusedResultId, pendingPrediction: this.pendingPrediction, resultChoice: this.resultChoice,
      angleChoice: this.angleChoice, ruleDrafts: [...this.ruleDrafts],
    };
  }

  get results(): ResultMessage[] {
    return this.messages.filter((message): message is ResultMessage => message.kind === 'result');
  }

  get hasContent() {
    return Boolean(this.messages.length || this.customChatTitle || this.composer.trim());
  }

  get chatTitle() {
    if (this.customChatTitle) return this.customChatTitle;
    const first = this.results[0];
    return first ? first.data.room.id + ' · ' + displayedDate(first.data.day, this.workspace.uiLocale)
      : text(this.workspace.uiLocale, 'untitledChat');
  }

  renameChat(title: string) {
    const name = title.trim();
    if (name) this.customChatTitle = name;
  }

  get exportCandidates() {
    return this.exportFilter === 'all' ? this.results : this.results.filter(result => result.bookmarked);
  }

  get selectedResults() {
    return this.results.filter(result => this.selectedResultIds.has(result.id));
  }

  get focusedResult(): ResultMessage | undefined {
    return this.findResult(this.focusedResultId) ?? this.results.at(-1);
  }

  findResult(id?: string | null): ResultMessage | undefined {
    return this.results.find(result => result.id === id);
  }

  get conversationLanguage() {
    return this.replyLanguage ?? this.workspace.uiLocale;
  }

  get conversationLocale() {
    return builtInLocale(this.replyLanguage, this.workspace.uiLocale);
  }

  private push(message: Omit<ResultMessage, 'id'> | Omit<ComparisonMessage, 'id'> | Omit<RankingMessage, 'id'>
    | Omit<Exclude<Message, ResultMessage | ComparisonMessage | RankingMessage>, 'id'>) {
    const id = 'message-' + ++this.messageId;
    this.messages.push({ ...message, id });
    if (message.kind === 'result') this.focusedResultId = id;
    return id;
  }

  private say(key: CopyKey, values?: Record<string, string | number>, kind: 'user' | 'assistant' | 'error' | 'clarify'
    | 'choose-result' | 'choose-angles' = 'assistant') {
    this.push({ kind, text: text(this.conversationLocale, key, values), language: this.conversationLocale });
  }

  closeEditors() {
    this.dismissRule();
    this.closeExport();
    this.reveal = null;
  }

  setComposer(value: string) {
    this.composer = value;
  }

  focusResult(result: ResultMessage) {
    this.focusedResultId = result.id;
    if (this.issues.length === 0) {
      this.draft = { ...conditionsOf(result.data), preference: result.selection.preference };
    }
  }

  private async task(work: () => Promise<void>, fallback: CopyKey) {
    if (this.busy) return;
    this.busy = true;
    try {
      await work();
    } catch (error) {
      runInAction(() => this.say(requestError(error, fallback), undefined, 'error'));
    } finally {
      runInAction(() => { this.busy = false; });
    }
  }

  async send(message = this.composer) {
    const query = message.trim();
    if (!query || this.busy) return;
    this.composer = '';
    this.resultChoice = null;
    this.angleChoice = null;
    this.push({ kind: 'user', text: query });
    await this.task(async () => {
      const all = this.results;
      const active = this.focusedResult;
      const recent = all.length <= 20 ? all
        : all.filter((result, index) => index < 1 || index >= all.length - 17
          || result.id === active?.id || all[index + 1]?.id === active?.id);
      const data = await api<ChatResponse>('chat', {
        message: query, draft: this.draft, language: this.conversationLanguage,
        active_result_id: active?.id,
        results: recent.map(result => ({ id: result.id, ...conditionsOf(result.data) })),
      });
      runInAction(() => {
        if (data.requested_language) this.replyLanguage = data.requested_language;
        this.interpretation = data.interpretation;
      });
      await this.execute(data, query);
    }, 'requestError');
  }

  private async execute(command: ChatResponse, question: string) {
    const { action, localized_reply: reply } = command;
    if (action === 'language' || action === 'help') {
      if (reply) this.push({ kind: 'assistant', text: reply, language: command.language });
      else this.say(isLocale(command.language) ? 'languageAck' : 'languageService');
      return;
    }
    if (command.unsupported_angles?.length) {
      this.say('supportedAngles');
      return;
    }
    if (action === 'rule' && command.rule_issue) {
      this.say('ruleCutoffHint');
      return;
    }
    if (action === 'rank' && command.ranking_issue) {
      this.say(command.ranking_issue === 'filter' ? 'rankingFilterHint' : 'rankingSortHint');
      return;
    }
    if (action === 'predict' || action === 'clarify') {
      const comparison = command.compare_scenario ? command
        : command.starts_new_prediction ? null : this.pendingPrediction;
      runInAction(() => {
        this.draft = command.draft;
        this.issues = command.issues;
        this.pendingPrediction = action === 'clarify' ? comparison : null;
        if (action === 'clarify') {
          if (reply) this.push({ kind: 'clarify', text: reply, language: command.language });
          else this.say('clarify', undefined, 'clarify');
        }
      });
      if (action === 'predict') await this.predictDraft(this.draft, comparison, reply);
      return;
    }

    const ids = command.result_choices?.length ? command.result_choices : command.against_choices;
    if (ids?.length) {
      runInAction(() => {
        this.resultChoice = { ids, command, question, role: command.result_choices?.length ? 'base' : 'against' };
        this.say('chooseResult', undefined, 'choose-result');
      });
      return;
    }
    if (command.reference_missed) {
      this.say('noMatchingResult');
      return;
    }
    const result = command.result_id ? this.findResult(command.result_id) : this.focusedResult;
    if (!result) {
      if (action === 'rule' && command.preference) {
        this.draft.preference = command.preference;
        this.say('nextPreference', { preference: text(this.conversationLocale, command.preference) });
      } else {
        this.say('needResult');
      }
      return;
    }
    this.focusResult(result);

    switch (action) {
      case 'explain': {
        const mentionsInspection = /this angle|(?:selected|highlighted|pinned) angle|这个角度|选中的角度|高亮角度/i.test(question)
          && !/recommend|推荐/i.test(question);
        const angles = command.angles?.length ? command.angles
          : mentionsInspection && result.inspectedAngle != null ? [result.inspectedAngle] : [];
        await this.explain(question, result, angles, command.metrics);
        break;
      }
      case 'rule': {
        const preference = command.preference;
        const hasCustom = result.customRule || result.selection.preference === 'custom';
        if (!preference || (preference === 'custom' && !hasCustom)) {
          this.openRule(result, command.rule);
          this.say('rulePreviewReady');
          this.showResult(result, 'editor');
        } else {
          this.choosePreference(result, preference);
          this.say('ruleUpdated');
          this.showResult(result);
        }
        break;
      }
      case 'compare': {
        const other = this.findResult(command.against_result_id);
        const angles = command.angles ?? [];
        if (other) this.compareScenarios(result, other, angles, command.metrics);
        else if (command.compare_cards) this.say('needTwoResults');
        else if (angles.length >= 2) this.compareAngles(result, angles, command.metrics);
        else if (command.compare_all && command.metrics?.length) this.compareAngles(result, ANGLES, command.metrics);
        else if (command.compare_all) {
          this.setValuesOpen(result, true);
          this.say('chartUpdated');
          this.showResult(result);
        } else {
          runInAction(() => {
            this.angleChoice = { resultId: result.id, angles, metrics: command.metrics };
            this.say('chooseCompareAngles', undefined, 'choose-angles');
          });
        }
        break;
      }
      case 'rank':
        this.push({ kind: 'ranking', data: result.data, angles: command.angles ?? [],
          filters: command.filters ?? [], sort: command.sort, language: this.replyLanguage ?? undefined });
        break;
      case 'chart':
        this.changeChart(result, command);
        break;
      case 'export':
        this.openExport(result, command.angles, command.format);
        break;
    }
  }

  async chooseResult(id: string) {
    const choice = this.resultChoice;
    if (!choice?.ids.includes(id) || this.busy) return;
    this.resultChoice = null;
    const command = choice.role === 'base'
      ? { ...choice.command, result_choices: [], result_id: id }
      : { ...choice.command, against_choices: [], against_result_id: id };
    await this.task(() => this.execute(command, choice.question), 'requestError');
  }

  toggleCompareAngle(angle: Angle) {
    if (!this.angleChoice) return;
    const { angles } = this.angleChoice;
    this.angleChoice.angles = angles.includes(angle) ? angles.filter(item => item !== angle) : [...angles, angle];
  }

  finishAngleChoice() {
    if (!this.angleChoice || this.angleChoice.angles.length < 2) return;
    const result = this.findResult(this.angleChoice.resultId);
    if (result) this.compareAngles(result, this.angleChoice.angles, this.angleChoice.metrics);
    this.angleChoice = null;
  }

  private compareAngles(result: ResultMessage, angles: Angle[], metrics?: MetricId[]) {
    const [first, ...others] = angles;
    this.push({ kind: 'comparison', left: result.data, right: result.data, metrics,
      pairs: others.map(angle => [first, angle]), language: this.replyLanguage ?? undefined });
  }

  private compareScenarios(base: ResultMessage, other: ResultMessage, angles: Angle[], metrics?: MetricId[]) {
    this.push({ kind: 'comparison', left: base.data, right: other.data, metrics,
      pairs: (angles.length ? angles : ANGLES).map(angle => [angle, angle]), language: this.replyLanguage ?? undefined });
  }

  async setClarification<K extends 'room_id' | 'day' | 'hour'>(field: K, value: Draft[K]) {
    if (this.busy) return;
    this.draft = { ...this.draft, [field]: value };
    if (value !== null) {
      const issueField = field === 'room_id' ? 'room' : field;
      this.issues = this.issues.filter(issue => issue.field !== issueField);
    }
    if (!getConditions(this.draft) || this.issues.length) return;
    const pending = this.pendingPrediction;
    this.pendingPrediction = null;
    await this.task(() => this.predictDraft(this.draft, pending), 'predictError');
  }

  private async predictDraft(draft: Draft, comparison?: ChatResponse | null,
                             introduction?: string | null, source?: ResultMessage) {
    const conditions = getConditions(draft);
    if (!conditions) return;
    const base = comparison ? this.findResult(comparison.result_id) : undefined;
    const inherited = source ?? base ?? this.focusedResult;
    const selection = this.selectionFor(draft.preference, inherited);
    const data = await api<Prediction>('predict', conditions);
    runInAction(() => {
      if (introduction) this.push({ kind: 'assistant', text: introduction, language: this.conversationLanguage });
      const id = this.push({ kind: 'result', data, selection, customRule: inherited?.customRule,
        language: this.replyLanguage ?? undefined });
      this.draft = { ...conditionsOf(data), preference: selection.preference };
      this.issues = [];
      this.conditionDraft = null;
      this.ruleEditorId = null;
      const result = this.findResult(id);
      if (base && result) this.compareScenarios(base, result, comparison?.angles ?? [], comparison?.metrics);
      if (source && result) this.showResult(result, 'header');
    });
  }

  private selectionFor(preference: Preference | null, source?: ResultMessage): RuleSelection {
    return chooseRule(preference, source?.customRule ?? source?.selection.rule);
  }

  private async explain(question: string, result: ResultMessage, angles?: Angle[], metrics?: MetricId[]) {
    const { selection } = result;
    const detail = await api<{ text: string; language: string }>('explain', {
      conditions: conditionsOf(result.data),
      rule: selection.preference && validRule(selection.rule) ? selection.rule : null,
      question, language: this.conversationLanguage, angles: angles ?? [], metrics: metrics ?? [],
    });
    runInAction(() => this.push({ kind: 'assistant', text: detail.text, language: detail.language }));
  }

  askAbout(result: ResultMessage) {
    if (this.busy) return;
    this.focusResult(result);
    const question = text(this.conversationLocale, result.selection.preference ? 'explainRecommended' : 'explainCompare');
    this.push({ kind: 'user', text: question });
    void this.task(() => this.explain(question, result), 'explainError');
  }

  choosePreference(result: ResultMessage, preference: Preference | null) {
    if (preference === 'custom' && !result.customRule && result.selection.preference !== 'custom') {
      this.openRule(result);
      return;
    }
    result.selection = this.selectionFor(preference, result);
    this.focusResult(result);
  }

  openConditions() {
    this.conditionDraft = { ...this.draft };
  }

  setCondition<K extends keyof Draft>(field: K, value: Draft[K]) {
    if (this.conditionDraft) this.conditionDraft[field] = value;
  }

  runConditions() {
    if (this.conditionDraft) this.startScenario(this.conditionDraft);
  }

  closeConditions() {
    this.conditionDraft = null;
  }

  changeConditions(result: ResultMessage, next: Conditions) {
    const original = conditionsOf(result.data);
    if (next.room_id === original.room_id && next.day === original.day && next.hour === original.hour) return;
    this.startScenario({ ...next, preference: result.selection.preference }, result);
  }

  private startScenario(draft: Draft, source?: ResultMessage) {
    const conditions = getConditions(draft);
    if (!conditions || this.busy) return;
    this.say('manualPrediction', {
      room: conditions.room_id, date: displayedDate(conditions.day, this.conversationLocale),
      hour: displayedHour(conditions.hour),
    }, 'user');
    this.pendingPrediction = null;
    this.resultChoice = null;
    this.angleChoice = null;
    void this.task(() => this.predictDraft(draft, null, null, source), 'predictError');
  }

  openRule(result: ResultMessage, patch?: RuleEdit) {
    this.focusResult(result);
    const draft = this.ruleDrafts.get(result.id);
    if (!draft || patch) this.ruleDrafts.set(result.id, editRule(draft ?? result.customRule ?? result.selection.rule, patch ?? {}));
    this.ruleEditorId = result.id;
  }

  setRule(rule: Rule) {
    if (this.ruleEditorId) this.ruleDrafts.set(this.ruleEditorId, rule);
  }

  applyRule() {
    if (this.busy || !this.ruleEditorId) return;
    const rule = this.ruleDrafts.get(this.ruleEditorId);
    const result = this.findResult(this.ruleEditorId);
    if (!rule || !result || !validRule(rule)) return;
    result.customRule = { ...rule };
    this.choosePreference(result, 'custom');
    this.cancelRule();
  }

  dismissRule() {
    this.ruleEditorId = null;
  }

  cancelRule() {
    if (this.ruleEditorId) this.ruleDrafts.delete(this.ruleEditorId);
    this.dismissRule();
  }

  private showResult(result: ResultMessage, target: NonNullable<ChatSession['reveal']>['target'] = 'charts') {
    this.reveal = { resultId: result.id, target };
  }

  inspect(result: ResultMessage, angle: Angle | null) {
    result.inspectedAngle = angle;
    this.focusResult(result);
  }

  toggleMetric(result: ResultMessage, metric: MetricId) {
    const hidden = result.hiddenMetrics ?? [];
    result.hiddenMetrics = hidden.includes(metric) ? hidden.filter(id => id !== metric) : [...hidden, metric];
    this.focusResult(result);
  }

  setValuesOpen(result: ResultMessage, open: boolean) {
    result.valuesOpen = open;
  }

  private changeChart(result: ResultMessage, command: ChatResponse) {
    const metrics = command.metrics ?? [];
    const angle = command.angles?.[0];
    if (!metrics.length && angle === undefined && command.table_open == null) {
      this.say('chartOptions');
      return;
    }
    if (metrics.length) {
      const hidden = result.hiddenMetrics ?? [];
      result.hiddenMetrics = command.visibility === 'only' ? METRIC_IDS.filter(id => !metrics.includes(id))
        : command.visibility === 'hide' ? [...new Set([...hidden, ...metrics])]
          : hidden.filter(id => !metrics.includes(id));
    }
    if (angle !== undefined) this.inspect(result, angle);
    if (command.table_open != null) this.setValuesOpen(result, command.table_open);
    this.say('chartUpdated');
    this.showResult(result);
  }

  toggleBookmark(result: ResultMessage) {
    result.bookmarked = !result.bookmarked;
  }

  setExportFilter(filter: ChatSession['exportFilter']) {
    this.exportFilter = filter;
  }

  selectResult(result: ResultMessage, selected: boolean) {
    if (selected) this.selectedResultIds.add(result.id);
    else this.selectedResultIds.delete(result.id);
  }

  selectVisibleResults(selected: boolean) {
    for (const result of this.exportCandidates) this.selectResult(result, selected);
  }

  clearSelection() {
    this.selectedResultIds.clear();
  }

  openExport(result: ResultMessage, angles?: Angle[], format?: ExportTarget['format']) {
    this.focusResult(result);
    this.exportTarget = { prediction: result.data, selection: result.selection, angles, format };
    this.exportLanguage = this.conversationLanguage;
  }

  closeExport() {
    this.exportTarget = null;
    this.exportLanguage = null;
  }
}

export class Workspace {
  rooms: Room[] = [];
  health: Health | null = null;
  connected = true;
  uiLocale = storedLocale();
  storageUnavailable = false;
  private chats = new Map<string, ChatSession>();
  private activeId = '';

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });
    this.restoreChats();
  }

  get activeChat() {
    return this.chats.get(this.activeId)!;
  }

  get recentChats() {
    return [...this.chats.values()].filter(chat => chat.id === this.activeId || chat.hasContent)
      .sort((left, right) => right.updatedAt - left.updatedAt);
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

  newChat() {
    if (!this.activeChat.hasContent && !this.activeChat.conditionDraft) return;
    const chat = new ChatSession(this, this.newChatId());
    this.track(chat);
    this.openChat(chat.id);
  }

  openChat(id: string) {
    if (id === this.activeId || !this.chats.has(id)) return;
    this.chats.get(this.activeId)?.closeEditors();
    this.activeId = id;
    this.writeStorage(ACTIVE_CHAT_STORAGE, id);
  }

  setLocale(locale: Locale) {
    this.uiLocale = locale;
    this.activeChat.replyLanguage = null;
    this.writeStorage(LANGUAGE_STORAGE, locale);
  }

  private newChatId() {
    return globalThis.crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  private restoreChats() {
    let active: string | null = null;
    try {
      const storage = window.localStorage;
      active = storage.getItem(ACTIVE_CHAT_STORAGE);
      for (let i = 0; i < storage.length; i++) {
        const key = storage.key(i);
        if (!key?.startsWith(CHAT_STORAGE)) continue;
        const id = key.slice(CHAT_STORAGE.length);
        try {
          const saved: SavedChat = JSON.parse(storage.getItem(key) ?? 'null');
          const snapshot: ChatSnapshot = JSON.parse(saved.snapshot);
          if (!Array.isArray(snapshot.messages) || !Number.isFinite(saved.updatedAt)) continue;
          this.track(new ChatSession(this, id, saved.updatedAt, snapshot));
        } catch {
          // A damaged record should not prevent the other chats from opening.
        }
      }
    } catch {
      this.storageUnavailable = true;
    }
    const id = active || this.recentChats[0]?.id || this.newChatId();
    if (!this.chats.has(id)) this.track(new ChatSession(this, id));
    this.openChat(id);
  }

  private track(chat: ChatSession) {
    this.chats.set(chat.id, chat);
    reaction(() => JSON.stringify(chat.snapshot), snapshot => {
      if (chat.hasContent) {
        chat.updatedAt = Date.now();
        this.writeStorage(CHAT_STORAGE + chat.id, JSON.stringify({ updatedAt: chat.updatedAt, snapshot }));
      } else {
        this.writeStorage(CHAT_STORAGE + chat.id, null);
      }
    });
  }

  private writeStorage(key: string, value: string | null) {
    try {
      if (value === null) window.localStorage.removeItem(key);
      else window.localStorage.setItem(key, value);
    } catch {
      this.storageUnavailable = true;
    }
  }
}
