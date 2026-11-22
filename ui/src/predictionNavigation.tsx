import { Fragment, useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { ArrowUpRight, Bookmark, Download, X } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import { useI18n } from './i18n';
import { downloadPredictions } from './predictionExport';
import { rank } from './scoring';
import type { ChatSession } from './store';
import { displayedHour, type ResultMessage } from './types';

export function BookmarkButton({ bookmarked, onClick }: { bookmarked: boolean; onClick: () => void }) {
  const { t } = useI18n();
  return <button type="button" className="icon-button bookmark-button" aria-label={t('bookmarkPrediction')}
    aria-pressed={bookmarked} title={t(bookmarked ? 'removeBookmark' : 'bookmarkPrediction')} onClick={onClick}>
    <Bookmark size={17} fill={bookmarked ? 'currentColor' : 'none'} aria-hidden="true" />
  </button>;
}

const RailPreview = observer(function RailPreview({ result, id, panel, onNavigate, onBookmark, onEnter, onLeave }: {
  result: ResultMessage;
  id: string;
  panel: RefObject<HTMLDivElement | null>;
  onNavigate: (keyboard: boolean) => void;
  onBookmark: () => void;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const { date, t } = useI18n();
  const recommended = rank(result.data.rows, result.selection).filter(row => row.recommended);
  return <div className="rail-preview" id={id} role="dialog" aria-label={t('predictionPreview')} ref={panel}
    onPointerEnter={onEnter} onPointerLeave={onLeave} onFocusCapture={onEnter} onBlur={event => {
      if (!event.currentTarget.contains(event.relatedTarget)) onLeave();
    }}>
    <div className="rail-preview-top">
      <button type="button" className="rail-preview-title" aria-label={t('viewPrediction', { room: result.data.room.id })}
        onClick={event => onNavigate(event.detail === 0)}>
        {result.data.room.id}<ArrowUpRight size={14} aria-hidden="true" />
      </button>
      <BookmarkButton bookmarked={Boolean(result.bookmarked)} onClick={onBookmark} />
    </div>
    <p>{date(result.data.day)} · {displayedHour(result.data.hour)}</p>
    <p>{t(result.selection.preference ?? 'noPreference')}
      {recommended.length > 0 && <span> · {t('recommended')} {recommended.map(row => row.PVSD_Angle_deg + '°').join(', ')}</span>}</p>
  </div>;
});

export const PredictionRail = observer(function PredictionRail({ store, conversation }: {
  store: ChatSession;
  conversation: RefObject<HTMLDivElement | null>;
}) {
  const { date, t } = useI18n();
  const id = useId();
  const results = store.results;
  const [visibleId, setVisibleId] = useState<string | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const rail = useRef<HTMLElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const markers = useRef(new Map<string, HTMLButtonElement>());
  const closeTimer = useRef<number | undefined>(undefined);
  const tap = useRef(false);
  const previewed = results.find(result => result.id === previewId);
  const active = visibleId ?? store.focusedResultId;

  const cancelClose = useCallback(() => window.clearTimeout(closeTimer.current), []);

  useEffect(() => cancelClose, [cancelClose]);

  useLayoutEffect(() => {
    const scroller = conversation.current;
    if (!scroller || !results.length) return;
    let frame = 0;
    function update() {
      if (!scroller) return;
      const viewport = scroller.getBoundingClientRect();
      const reference = viewport.top + viewport.height * 0.25;
      let nearest: string | null = null;
      let distance = Infinity;
      for (const { id } of results) {
        const bounds = document.getElementById(id)?.getBoundingClientRect();
        if (!bounds) continue;
        const gap = Math.max(bounds.top - reference, reference - bounds.bottom, 0);
        if (gap >= distance) continue;
        nearest = id;
        distance = gap;
        if (!gap) break;
      }
      setVisibleId(nearest);
    }
    function schedule() {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(update);
    }
    update();
    scroller.addEventListener('scroll', schedule, { passive: true });
    const observer = new ResizeObserver(schedule);
    observer.observe(scroller);
    if (scroller.firstElementChild) observer.observe(scroller.firstElementChild);
    return () => {
      scroller.removeEventListener('scroll', schedule);
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [conversation, results]);

  useEffect(() => {
    const marker = active && markers.current.get(active);
    const list = track.current;
    if (!marker || !list) return;
    const item = marker.getBoundingClientRect();
    const bounds = list.getBoundingClientRect();
    if (item.top < bounds.top) list.scrollTop += item.top - bounds.top;
    else if (item.bottom > bounds.bottom) list.scrollTop += item.bottom - bounds.bottom;
  }, [active]);

  useLayoutEffect(() => {
    if (!previewed) return;
    const marker = markers.current.get(previewed.id);
    const popup = panel.current;
    const list = track.current;
    if (!marker || !popup || !list) return;
    function position() {
      if (!marker || !popup || !list) return;
      const anchor = marker.getBoundingClientRect();
      const bounds = list.getBoundingClientRect();
      if (anchor.bottom < bounds.top || anchor.top > bounds.bottom) {
        setPreviewId(null);
        return;
      }
      const pane = rail.current?.getBoundingClientRect();
      const top = Math.max(8, (pane?.top ?? 0) + 6);
      const bottom = Math.min(window.innerHeight - 8, (pane?.bottom ?? window.innerHeight) - 6);
      const centered = anchor.top + (anchor.height - popup.offsetHeight) / 2;
      popup.style.left = Math.max(8, Math.min(anchor.right + 5, window.innerWidth - popup.offsetWidth - 8)) + 'px';
      popup.style.top = Math.max(top, Math.min(centered, bottom - popup.offsetHeight)) + 'px';
      popup.style.visibility = 'visible';
    }
    function dismiss(event: PointerEvent) {
      if (event.target instanceof Node && !rail.current?.contains(event.target)) setPreviewId(null);
    }
    function escape(event: KeyboardEvent) {
      if (event.key !== 'Escape') return;
      if (popup?.contains(document.activeElement)) marker?.focus({ preventScroll: true });
      setPreviewId(null);
    }
    position();
    list.addEventListener('scroll', position, { passive: true });
    window.addEventListener('resize', position);
    const observer = new ResizeObserver(position);
    observer.observe(popup);
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      observer.disconnect();
      list.removeEventListener('scroll', position);
      window.removeEventListener('resize', position);
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [previewed]);

  function open(id: string) {
    cancelClose();
    setPreviewId(id);
  }

  function leave() {
    cancelClose();
    closeTimer.current = window.setTimeout(() => {
      if (!panel.current?.contains(document.activeElement)) setPreviewId(null);
    }, 180);
  }

  function navigate(result: ResultMessage, keyboard: boolean) {
    cancelClose();
    setPreviewId(null);
    store.focusResult(result);
    const card = document.getElementById(result.id);
    card?.scrollIntoView({ block: 'start' });
    if (keyboard) card?.focus({ preventScroll: true });
  }

  if (!results.length) return null;
  return <nav className="prediction-rail" ref={rail} aria-label={t('predictionOverview')}>
      <div className="rail-track" ref={track}>
        {results.map(result => <Fragment key={result.id}>
          <button type="button" className="rail-marker" ref={node => {
            if (node) markers.current.set(result.id, node);
            else markers.current.delete(result.id);
          }} aria-label={t('predictionMarker', {
            room: result.data.room.id, date: date(result.data.day), hour: displayedHour(result.data.hour),
          })} aria-current={active === result.id ? 'location' : undefined} aria-controls={previewId === result.id ? id : result.id}
            aria-haspopup="dialog" aria-expanded={previewId === result.id}
            data-bookmarked={Boolean(result.bookmarked)}
            onPointerEnter={event => { if (event.pointerType === 'mouse') open(result.id); }}
            onPointerLeave={leave}
            onPointerDown={event => { tap.current = event.pointerType !== 'mouse'; }}
            onFocus={event => { if (!tap.current && event.currentTarget.matches(':focus-visible')) open(result.id); }}
            onBlur={leave}
            onClick={event => {
              if (tap.current && event.detail > 0) {
                cancelClose();
                setPreviewId(current => current === result.id ? null : result.id);
              } else navigate(result, event.detail === 0);
              tap.current = false;
            }}><span aria-hidden="true" /></button>
          {previewed?.id === result.id && <RailPreview result={result} id={id} panel={panel}
            onNavigate={keyboard => navigate(result, keyboard)} onBookmark={() => store.toggleBookmark(result)}
            onEnter={cancelClose} onLeave={leave} />}
        </Fragment>)}
      </div>
  </nav>;
});

export const ChatExportButton = observer(function ChatExportButton({ store }: { store: ChatSession }) {
  const { t } = useI18n();
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);

  if (!store.results.length) return null;
  return <>
    <button ref={button} type="button" className="icon-button chat-export" title={t('exportChatPredictions')}
      aria-label={t('exportChatPredictions')} aria-haspopup="dialog" aria-expanded={open}
      aria-controls={open ? id : undefined} onClick={() => setOpen(!open)}>
      <Download size={17} aria-hidden="true" />
    </button>
    {open && <PredictionExportPanel store={store} id={id} anchor={button} onClose={close} />}
  </>;
});

const PredictionExportPanel = observer(function PredictionExportPanel({ store, id, anchor, onClose }: {
  store: ChatSession;
  id: string;
  anchor: RefObject<HTMLButtonElement | null>;
  onClose: () => void;
}) {
  const { date, number, t } = useI18n();
  const title = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const results = store.exportCandidates;
  const selected = store.selectedResults;
  const allVisibleSelected = results.length > 0 && results.every(result => store.selectedResultIds.has(result.id));
  const showClearSelection = selected.length > 0 && (!allVisibleSelected || selected.length > results.length);
  useLayoutEffect(() => {
    const panel = dialog.current;
    const button = anchor.current;
    if (!panel || !button) return;
    const narrow = window.matchMedia('(max-width: 700px)');
    const close = () => panel.close();

    function position() {
      if (!panel || !button || narrow.matches) return;
      const frame = button.getBoundingClientRect();
      const top = frame.bottom + 8;
      const left = frame.left + (frame.width - panel.offsetWidth) / 2;
      panel.style.maxHeight = Math.max(0, window.innerHeight - top - 8) + 'px';
      panel.style.left = Math.max(8, Math.min(left, window.innerWidth - panel.offsetWidth - 8)) + 'px';
      panel.style.top = top + 'px';
    }
    function dismiss(event: PointerEvent) {
      if (event.target instanceof Node && !panel?.contains(event.target) && !button?.contains(event.target)) close();
    }
    function escape(event: KeyboardEvent) {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      event.preventDefault();
      close();
      button?.focus({ preventScroll: true });
    }

    if (narrow.matches) panel.showModal();
    else {
      panel.show();
      position();
      document.addEventListener('pointerdown', dismiss);
      document.addEventListener('keydown', escape);
    }
    const observer = new ResizeObserver(position);
    observer.observe(panel);
    if (button.parentElement) observer.observe(button.parentElement);
    window.addEventListener('resize', position);
    narrow.addEventListener('change', close);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', position);
      narrow.removeEventListener('change', close);
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, [anchor]);

  return createPortal(
    <dialog id={id} className="prediction-export-dialog prediction-export-panel" ref={dialog} onClose={onClose}
      aria-labelledby={title} onClick={event => {
        if (event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right
          || event.clientY < bounds.top || event.clientY > bounds.bottom) event.currentTarget.close();
      }}>
      <div className="prediction-export-heading">
        <h2 id={title}>{t('exportPredictions')}</h2>
        <button type="button" className="icon-button" aria-label={t('close')} onClick={() => {
          dialog.current?.close();
          anchor.current?.focus({ preventScroll: true });
        }}><X size={18} aria-hidden="true" /></button>
      </div>
      <div className="prediction-filters" role="group" aria-label={t('filterPredictions')}>
        <button type="button" aria-pressed={store.exportFilter === 'bookmarked'} onClick={() => store.setExportFilter('bookmarked')}>
          <Bookmark size={14} aria-hidden="true" />{t('bookmarkedPredictions')}
        </button>
        <button type="button" aria-pressed={store.exportFilter === 'all'} onClick={() => store.setExportFilter('all')}>
          {t('allPredictions')}
        </button>
      </div>
      {results.length > 0 ? <>
        <div className="prediction-selection-actions">
          <button type="button" className="text-button" onClick={() => store.selectVisibleResults(!allVisibleSelected)}>
            {t(allVisibleSelected ? 'deselectShown' : 'selectShown')}
          </button>
          {showClearSelection && <button type="button" className="text-button" onClick={store.clearSelection}>{t('clearSelection')}</button>}
        </div>
        <ul className="prediction-selection">
          {results.map(result => <li key={result.id}>
            <label className={store.selectedResultIds.has(result.id) ? 'selected' : undefined}>
              <input type="checkbox" checked={store.selectedResultIds.has(result.id)}
                onChange={event => store.selectResult(result, event.target.checked)} />
              <span><strong>{result.data.room.id}</strong><small>{date(result.data.day)} · {displayedHour(result.data.hour)}</small></span>
              {result.bookmarked && <Bookmark className="selection-bookmark" size={13} fill="currentColor" aria-hidden="true" />}
            </label>
          </li>)}
        </ul>
      </> : <p className="prediction-selection-empty">{t(store.results.length ? 'emptyBookmarks' : 'emptyHistory')}</p>}
      <div className="prediction-export-actions">
        <div className="prediction-export-status" aria-live="polite">
          {t('selectedPredictions', { count: number(selected.length) })}
          {results.length === 0 && selected.length > 0 && <button type="button" className="text-button" onClick={store.clearSelection}>{t('clearSelection')}</button>}
        </div>
        <div className="prediction-formats">
          <button type="button" className="button secondary" aria-label={t('csv')} disabled={!selected.length}
            onClick={() => downloadPredictions('csv', selected)}>CSV</button>
          <button type="button" className="button secondary" aria-label={t('json')} disabled={!selected.length}
            onClick={() => downloadPredictions('json', selected)}>JSON</button>
        </div>
      </div>
    </dialog>, document.body);
});
