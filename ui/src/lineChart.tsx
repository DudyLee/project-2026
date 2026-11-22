import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { useI18n } from './i18n';
import { tooltipPosition, type Point } from './tooltipPlacement';
import type { Angle } from './types';

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  values: (number | null)[];
  details: string[];
  dashed?: boolean;
}

type LegendItem = Pick<ChartSeries, 'id' | 'label' | 'color' | 'dashed'>;

export interface ChartAnchor {
  button: HTMLButtonElement;
  position: Point;
  touch: boolean;
}

export interface ChartInspection {
  angle: Angle;
  anchor: ChartAnchor;
  followPointer: boolean;
}

interface LineChartProps {
  title: string;
  note: ReactNode;
  controls?: ReactNode;
  legend?: LegendItem[];
  renderTooltip?: (index: number) => ReactNode;
  angles: Angle[];
  series: ChartSeries[];
  hidden: ReadonlySet<string>;
  active?: string | null;
  recommended: ReadonlySet<Angle>;
  hovered: Angle | null;
  pinned: Angle | null;
  inspection: ChartInspection | null;
  onLegendClick: (id: string) => void;
  onHover: (angle: Angle, anchor: ChartAnchor | null) => void;
  onPin: (angle: Angle, anchor: ChartAnchor) => void;
  onClear: () => void;
}

const TICKS = [100, 75, 50, 25, 0];
const INSET = 8;

function makeAnchor(button: HTMLButtonElement, pointer?: Point, touch = false): ChartAnchor {
  const rect = button.getBoundingClientRect();
  return {
    button, touch,
    position: pointer
      ? { x: (pointer.x - rect.left) / rect.width, y: (pointer.y - rect.top) / rect.height }
      : { x: 0.5, y: 0.5 },
  };
}

function Plot({ series, angles, active, hovered, pinned }: {
  series: ChartSeries[];
  angles: Angle[];
  active?: string | null;
  hovered: Angle | null;
  pinned: Angle | null;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const element = canvas;
    const selected = series.find(line => line.id === active);
    const ordered = series.filter(line => line !== selected);
    if (selected) ordered.push(selected);

    function draw() {
      const context = element.getContext('2d');
      const { width, height } = element.getBoundingClientRect();
      if (!context || !width || !height) return;

      const ratio = window.devicePixelRatio || 1;
      element.width = Math.round(width * ratio);
      element.height = Math.round(height * ratio);
      context.scale(ratio, ratio);
      const xAt = (index: number) => width * (index + 0.5) / angles.length;
      const yAt = (value: number) => INSET + (height - 2 * INSET) * (1 - value / 100);

      for (const line of ordered) {
        const emphasized = selected === line;
        context.globalAlpha = selected && !emphasized ? 0.48 : 1;
        context.lineJoin = 'round';
        context.lineCap = 'round';
        context.setLineDash(line.dashed ? [5, 4] : []);
        context.beginPath();
        line.values.forEach((value, index) => {
          if (value === null) return;
          if (index === 0 || line.values[index - 1] === null) context.moveTo(xAt(index), yAt(value));
          else context.lineTo(xAt(index), yAt(value));
        });
        if (emphasized) {
          context.lineWidth = 6;
          context.strokeStyle = '#f9fcfa';
          context.stroke();
        }
        context.lineWidth = emphasized ? 3.8 : 1.75;
        context.strokeStyle = line.color;
        context.stroke();

        context.setLineDash([]);
        context.fillStyle = line.dashed ? '#f9fcfa' : line.color;
        context.strokeStyle = line.dashed ? line.color : '#ffffff';
        context.lineWidth = 1;
        line.values.forEach((value, index) => {
          if (value === null) return;
          const inspected = angles[index] === hovered || angles[index] === pinned;
          const radius = inspected ? 5 : emphasized ? 4 : 3;
          context.beginPath();
          context.arc(xAt(index), yAt(value), radius, 0, 2 * Math.PI);
          context.fill();
          context.stroke();
        });
      }
    }

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(element);
    return () => observer.disconnect();
  }, [series, angles, active, hovered, pinned]);

  return <canvas ref={ref} className="line-canvas" aria-hidden="true" />;
}

export function SeriesSwatch({ color, dashed }: { color: string; dashed?: boolean }) {
  return <span aria-hidden="true" className="series-swatch" data-dashed={dashed} style={{ color }} />;
}

function ChartTooltip({ id, inspection, series, index, renderTooltip }: {
  id: string;
  inspection: ChartInspection;
  series: ChartSeries[];
  index: number;
  renderTooltip?: LineChartProps['renderTooltip'];
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<Point | null>(null);
  const { anchor: { button, position: offset }, followPointer } = inspection;

  useLayoutEffect(() => {
    const tooltip = ref.current;
    if (!tooltip) return;
    const scrollArea = button.closest('.conversation-scroll');
    let pointer: Point | null = null;
    let pending = 0;

    const update = () => {
      const anchor = button.getBoundingClientRect();
      const scrollBounds = scrollArea?.getBoundingClientRect();
      const top = Math.max(0, scrollBounds?.top ?? 0);
      const bottom = Math.min(window.innerHeight, scrollBounds?.bottom ?? window.innerHeight);

      if (anchor.bottom <= top || anchor.top >= bottom) {
        setPosition(null);
        return;
      }

      const cursor = pointer ?? { x: anchor.left + anchor.width * offset.x, y: anchor.top + anchor.height * offset.y };
      const next = tooltipPosition(cursor, { width: tooltip.offsetWidth, height: tooltip.offsetHeight },
        { width: document.documentElement.clientWidth, height: window.innerHeight });
      setPosition(previous => previous?.x === next.x && previous.y === next.y ? previous : next);
    };

    function schedule() {
      if (pending) return;
      pending = window.requestAnimationFrame(() => {
        pending = 0;
        update();
      });
    }

    function track(event: PointerEvent) {
      pointer = { x: event.clientX, y: event.clientY };
      schedule();
    }

    update();
    const observer = new ResizeObserver(schedule);
    observer.observe(tooltip);
    window.addEventListener('scroll', schedule, true);
    window.addEventListener('resize', schedule);
    if (followPointer) button.addEventListener('pointermove', track);
    return () => {
      observer.disconnect();
      window.removeEventListener('scroll', schedule, true);
      window.removeEventListener('resize', schedule);
      button.removeEventListener('pointermove', track);
      window.cancelAnimationFrame(pending);
    };
  }, [button, offset, followPointer]);

  return <div ref={ref} className={'line-tooltip' + (renderTooltip ? ' comparison-tooltip' : '')} id={id} role="tooltip"
    style={position ? { left: position.x, top: position.y } : { left: 0, top: 0, visibility: 'hidden' }}>
    <strong>{inspection.angle}°</strong>
    {renderTooltip ? renderTooltip(index) : <dl>{series.map(line => <div key={line.id}>
      <dt><SeriesSwatch color={line.color} dashed={line.dashed} />{line.label}</dt><dd>{line.details[index]}</dd>
    </div>)}</dl>}
  </div>;
}

export function LineChart({
  title, note, controls, angles, series, legend = series, renderTooltip, hidden, active, recommended, hovered, pinned, inspection,
  onLegendClick, onHover, onPin, onClear,
}: LineChartProps) {
  const { t } = useI18n();
  const id = useId();
  const visible = series.filter(line => !hidden.has(line.id));
  const tooltipIndex = inspection ? angles.indexOf(inspection.angle) : -1;

  return <section className={'chart-panel' + (renderTooltip ? ' comparison-chart' : '')} aria-labelledby={id}>
    <div className="chart-heading"><h3 className="line-chart-title" id={id}>{title}</h3>{controls}</div>
    <div className="chart-scale-note">{note}</div>
    <div className="line-chart">
      <div className="line-ticks" aria-hidden="true">{TICKS.map(tick => <span key={tick}>{tick}</span>)}</div>
      <div className="line-view">
        <div className="line-grid" aria-hidden="true">{TICKS.map(tick => <i key={tick} />)}</div>
        <Plot series={visible} angles={angles} active={active} hovered={hovered} pinned={pinned} />
        {!visible.length && <p className="line-empty">{t('showOneSeries')}</p>}
        <div className="angle-buttons" role="group" aria-label={t('selectAngles')}>
          {angles.map(angle => <button
            key={angle} type="button" className={'angle-button' + (recommended.has(angle) ? ' angle-recommended' : '')}
            aria-label={angle + '°' + (recommended.has(angle) ? ' · ' + t('recommended') : '')}
            aria-pressed={angle === pinned} data-highlighted={angle === (hovered ?? pinned)}
            aria-describedby={angle === inspection?.angle && visible.length ? id + '-tooltip' : undefined}
            onPointerEnter={event => {
              if (event.pointerType !== 'touch') {
                onHover(angle, makeAnchor(event.currentTarget, { x: event.clientX, y: event.clientY }));
              }
            }}
            onPointerLeave={event => { if (event.pointerType !== 'touch') onHover(angle, null); }}
            onFocus={event => {
              if (event.currentTarget.matches(':focus-visible')) onHover(angle, makeAnchor(event.currentTarget));
            }}
            onBlur={() => onHover(angle, null)}
            onClick={event => {
              const touch = event.nativeEvent instanceof PointerEvent && event.nativeEvent.pointerType === 'touch';
              onPin(angle, makeAnchor(event.currentTarget,
                event.detail ? { x: event.clientX, y: event.clientY } : undefined, touch));
            }}
            onKeyDown={event => {
              if (event.key === 'Escape') {
                event.preventDefault();
                onClear();
              }
            }}
          ><span>{angle}°</span></button>)}
        </div>
      </div>
    </div>
    {inspection && tooltipIndex >= 0 && visible.length > 0
      && <ChartTooltip id={id + '-tooltip'} inspection={inspection} series={visible} index={tooltipIndex} renderTooltip={renderTooltip} />}
    <div className="chart-legend" role="group" aria-label={t('chartLegend')}>
      {legend.map(line => <button type="button" key={line.id} className={'legend-button' + (line.id === active ? ' active-series' : '')}
          style={{ '--series-color': line.color } as CSSProperties} aria-current={line.id === active || undefined}
          aria-pressed={!hidden.has(line.id)} onClick={() => onLegendClick(line.id)}>
          <SeriesSwatch color={line.color} dashed={line.dashed} />{line.label}
        </button>)}
    </div>
  </section>;
}
