export interface Point {
  x: number;
  y: number;
}

interface Size {
  width: number;
  height: number;
}

const EDGE_GAP = 8;

export function tooltipPosition(cursor: Point, size: Size, viewport: Size): Point {
  return {
    x: Math.max(EDGE_GAP, Math.min(cursor.x + 12, viewport.width - size.width - EDGE_GAP)),
    y: Math.max(EDGE_GAP, Math.min(cursor.y + 6, viewport.height - size.height - EDGE_GAP)),
  };
}
