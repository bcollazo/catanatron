import { useEffect, useState, type MouseEvent } from "react";
import { Link } from "react-router-dom";
import { Button } from "@mui/material";
import { GridLoader } from "react-spinners";

import type { ReplayCatalogItem } from "../utils/api.types";
import { getReplayCatalog } from "../utils/apiClient";

import "./ReplayCatalogScreen.scss";

type StatRow = {
  label: string;
  value: string;
};

type Point = {
  x: number;
  y: number;
};

type CorrelationPoint = {
  x: number;
  y: number;
  count: number;
  label: string;
};

type CatalogFilters = {
  won: "all" | "yes" | "no";
  wentFirst: "all" | "yes" | "no";
  gameIndex?: number;
  openingPipDiffExact?: number;
  turnBucketStart?: number;
  turnBucketSize?: number;
};

function formatWentFirst(value: boolean | null) {
  if (value === null) return "N/A";
  return value ? "YES" : "NO";
}

function formatWon(value: boolean | null, winner: ReplayCatalogItem["winner"]) {
  if (value === null) return winner ? `WINNER=${winner}` : "N/A";
  return value ? "YES" : "NO";
}

function formatImportedAt(value: string | null) {
  if (!value) return "N/A";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function average(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function averageNullable(values: Array<number | null | undefined>) {
  const filtered = values.filter((v): v is number => typeof v === "number");
  return average(filtered);
}

function sum(values: number[]) {
  return values.reduce((acc, v) => acc + v, 0);
}

function pct(numerator: number, denominator: number) {
  if (denominator === 0) return 0;
  return (numerator / denominator) * 100;
}

function cumulativeWinRatePoints(rows: ReplayCatalogItem[]): Point[] {
  const ordered = [...rows].reverse(); // oldest -> newest
  let wins = 0;
  return ordered.map((row, index) => {
    if (row.won) wins += 1;
    return {
      x: index + 1,
      y: (wins / (index + 1)) * 100,
    };
  });
}

function rollingWinRatePoints(rows: ReplayCatalogItem[], windowSize: number): Point[] {
  const ordered = [...rows].reverse();
  const out: Point[] = [];
  for (let i = 0; i < ordered.length; i++) {
    const start = Math.max(0, i - windowSize + 1);
    const slice = ordered.slice(start, i + 1);
    const wins = slice.filter((row) => row.won).length;
    out.push({ x: i + 1, y: pct(wins, slice.length) });
  }
  return out;
}

function turnCountPoints(rows: ReplayCatalogItem[]): Point[] {
  const ordered = [...rows].reverse();
  return ordered.map((row, index) => ({ x: index + 1, y: row.turn_count }));
}

function vpDiffPoints(rows: ReplayCatalogItem[]): Point[] {
  const ordered = [...rows].reverse();
  return ordered.map((row, index) => {
    const us = row.us_final_vp ?? 0;
    const opp = row.opp_final_vp ?? 0;
    return { x: index + 1, y: us - opp };
  });
}

function openingPipDiffPoints(rows: ReplayCatalogItem[]): Point[] {
  const ordered = [...rows].reverse();
  return ordered
    .map((row, index) => ({ x: index + 1, y: row.opening_pip_diff }))
    .filter((point): point is Point => typeof point.y === "number");
}

function pipDiffWinRateCorrelation(rows: ReplayCatalogItem[]): CorrelationPoint[] {
  const buckets = new Map<number, { wins: number; total: number }>();
  for (const row of rows) {
    if (row.opening_pip_diff === null || row.opening_pip_diff === undefined) continue;
    const key = Math.trunc(row.opening_pip_diff);
    const value = buckets.get(key) ?? { wins: 0, total: 0 };
    value.total += 1;
    if (row.won) value.wins += 1;
    buckets.set(key, value);
  }
  return Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([diff, stats]) => ({
      x: diff,
      y: pct(stats.wins, stats.total),
      count: stats.total,
      label: `diff=${diff}`,
    }));
}

function turnBucketWinRate(rows: ReplayCatalogItem[], bucketSize = 50): CorrelationPoint[] {
  const buckets = new Map<number, { wins: number; total: number }>();
  for (const row of rows) {
    const bucketStart = Math.floor(row.turn_count / bucketSize) * bucketSize;
    const value = buckets.get(bucketStart) ?? { wins: 0, total: 0 };
    value.total += 1;
    if (row.won) value.wins += 1;
    buckets.set(bucketStart, value);
  }
  return Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([bucketStart, stats]) => ({
      x: bucketStart,
      y: pct(stats.wins, stats.total),
      count: stats.total,
      label: `${bucketStart}-${bucketStart + bucketSize - 1}`,
    }));
}

function firstCityPoints(rows: ReplayCatalogItem[]): Point[] {
  const ordered = [...rows].reverse();
  return ordered
    .map((row, index) => ({ x: index + 1, y: row.us_first_city_turn }))
    .filter((point): point is Point => typeof point.y === "number");
}

type MixPoint = {
  x: number;
  build: number;
  trade: number;
  dev: number;
  robber: number;
};

function actionMixPoints(rows: ReplayCatalogItem[]): MixPoint[] {
  const ordered = [...rows].reverse();
  return ordered.map((row, index) => {
    const total = row.us_action_total || 1;
    return {
      x: index + 1,
      build: pct(row.us_action_build, total),
      trade: pct(row.us_action_trade, total),
      dev: pct(row.us_action_dev, total),
      robber: pct(row.us_action_robber, total),
    };
  });
}

function splitBehaviorWinRate(
  rows: ReplayCatalogItem[],
  valueSelector: (row: ReplayCatalogItem) => number
) {
  const values = rows.map(valueSelector).sort((a, b) => a - b);
  const median = values[Math.floor(values.length / 2)] ?? 0;
  const high = rows.filter((row) => valueSelector(row) >= median);
  const low = rows.filter((row) => valueSelector(row) < median);
  const highWins = high.filter((row) => row.won).length;
  const lowWins = low.filter((row) => row.won).length;
  return {
    median,
    highWinRate: pct(highWins, high.length),
    lowWinRate: pct(lowWins, low.length),
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function InteractiveLineChart({
  points,
  yLabel,
  color = "#4da3ff",
  yMin,
  yMax,
  formatY = (v) => v.toFixed(1),
  onPointClick,
}: {
  points: Point[];
  yLabel: string;
  color?: string;
  yMin?: number;
  yMax?: number;
  formatY?: (value: number) => string;
  onPointClick?: (point: Point) => void;
}) {
  const width = 320;
  const height = 150;
  const paddingLeft = 34;
  const paddingRight = 10;
  const paddingY = 14;
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (points.length === 0) {
    return <div className="chart-empty">No data</div>;
  }

  const localYMin = yMin ?? Math.min(...points.map((p) => p.y));
  const localYMax = yMax ?? Math.max(...points.map((p) => p.y));
  const safeMin = localYMin === localYMax ? localYMin - 1 : localYMin;
  const safeMax = localYMin === localYMax ? localYMax + 1 : localYMax;
  const maxX = Math.max(...points.map((p) => p.x), 1);
  const spanX = width - paddingLeft - paddingRight;
  const spanY = height - paddingY * 2;
  const xAt = (x: number) => paddingLeft + (spanX * x) / maxX;
  const yAt = (y: number) =>
    height - paddingY - (spanY * (y - safeMin)) / (safeMax - safeMin);

  const toPath = points
    .map((p, i) => {
      const x = xAt(p.x);
      const y = yAt(p.y);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  const activeIndex = hoverIndex ?? points.length - 1;
  const activePoint = points[activeIndex];
  const activeX = xAt(activePoint.x);
  const activeY = yAt(activePoint.y);

  const handleMouseMove = (evt: MouseEvent<SVGSVGElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const relativeX = clamp(evt.clientX - rect.left, paddingLeft, width - paddingRight);
    const ratio = (relativeX - paddingLeft) / spanX;
    const idx = clamp(Math.round(ratio * (points.length - 1)), 0, points.length - 1);
    setHoverIndex(idx);
  };

  const handleMouseLeave = () => setHoverIndex(null);

  return (
    <div className="interactive-chart">
      <div className="chart-readout">
        <strong>{yLabel}:</strong> {formatY(activePoint.y)} (Game {activePoint.x})
      </div>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <line
          x1={paddingLeft}
          y1={height - paddingY}
          x2={width - paddingRight}
          y2={height - paddingY}
          stroke="#505050"
        />
        <line
          x1={paddingLeft}
          y1={paddingY}
          x2={paddingLeft}
          y2={height - paddingY}
          stroke="#505050"
        />
        <line
          x1={paddingLeft}
          y1={yAt((safeMin + safeMax) / 2)}
          x2={width - paddingRight}
          y2={yAt((safeMin + safeMax) / 2)}
          stroke="#2f2f2f"
        />
        <text x={4} y={paddingY + 4} fill="#aaaaaa" fontSize="10">
          {formatY(safeMax)}
        </text>
        <text x={4} y={yAt((safeMin + safeMax) / 2) + 4} fill="#8f8f8f" fontSize="10">
          {formatY((safeMin + safeMax) / 2)}
        </text>
        <text x={4} y={height - paddingY + 4} fill="#aaaaaa" fontSize="10">
          {formatY(safeMin)}
        </text>
        <path d={toPath} fill="none" stroke={color} strokeWidth={2.5} />
        {points.map((p, i) => (
          <circle
            key={`${p.x}-${i}`}
            cx={xAt(p.x)}
            cy={yAt(p.y)}
            r={i === activeIndex ? 4 : 2.3}
            fill={color}
            opacity={i === activeIndex ? 1 : 0.8}
            onClick={() => onPointClick?.(p)}
            style={{ cursor: onPointClick ? "pointer" : "default" }}
          />
        ))}
        <line x1={activeX} y1={paddingY} x2={activeX} y2={height - paddingY} stroke="#5d5d5d" strokeDasharray="3 3" />
        <circle cx={activeX} cy={activeY} r={4} fill={color} stroke="#111111" strokeWidth={1.5} pointerEvents="none" />
      </svg>
    </div>
  );
}

function SplitBarChart({
  firstWinRate,
  secondWinRate,
  firstCount,
  secondCount,
}: {
  firstWinRate: number;
  secondWinRate: number;
  firstCount: number;
  secondCount: number;
}) {
  const firstHeight = Math.max(2, firstWinRate);
  const secondHeight = Math.max(2, secondWinRate);
  return (
    <div className="split-bars">
      <div className="bar-column">
        <div className="bar-label">First ({firstCount})</div>
        <div className="bar-track">
          <div className="bar-fill first" style={{ height: `${firstHeight}%` }} />
        </div>
        <strong>{firstWinRate.toFixed(1)}%</strong>
      </div>
      <div className="bar-column">
        <div className="bar-label">Second ({secondCount})</div>
        <div className="bar-track">
          <div className="bar-fill second" style={{ height: `${secondHeight}%` }} />
        </div>
        <strong>{secondWinRate.toFixed(1)}%</strong>
      </div>
    </div>
  );
}

function ActionMixChart({
  points,
  onPointClick,
}: {
  points: MixPoint[];
  onPointClick?: (point: MixPoint) => void;
}) {
  const width = 320;
  const height = 150;
  const paddingLeft = 34;
  const paddingRight = 10;
  const paddingY = 14;
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  if (points.length === 0) {
    return <div className="chart-empty">No data</div>;
  }
  const maxX = Math.max(...points.map((p) => p.x), 1);
  const spanX = width - paddingLeft - paddingRight;
  const spanY = height - paddingY * 2;
  const xAt = (x: number) => paddingLeft + (spanX * x) / maxX;
  const yAt = (y: number) => height - paddingY - (spanY * y) / 100;

  const buildPath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(p.x).toFixed(2)} ${yAt(p.build).toFixed(2)}`)
    .join(" ");
  const tradePath = points
    .map((p, i) =>
      `${i === 0 ? "M" : "L"} ${xAt(p.x).toFixed(2)} ${yAt(p.build + p.trade).toFixed(2)}`
    )
    .join(" ");
  const devPath = points
    .map((p, i) =>
      `${i === 0 ? "M" : "L"} ${xAt(p.x).toFixed(2)} ${yAt(p.build + p.trade + p.dev).toFixed(2)}`
    )
    .join(" ");
  const robberPath = points
    .map((p, i) =>
      `${
        i === 0 ? "M" : "L"
      } ${xAt(p.x).toFixed(2)} ${yAt(p.build + p.trade + p.dev + p.robber).toFixed(2)}`
    )
    .join(" ");

  const activeIndex = hoverIndex ?? points.length - 1;
  const active = points[activeIndex];
  const activeX = xAt(active.x);

  const handleMouseMove = (evt: MouseEvent<SVGSVGElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const relativeX = clamp(evt.clientX - rect.left, paddingLeft, width - paddingRight);
    const ratio = (relativeX - paddingLeft) / spanX;
    const idx = clamp(Math.round(ratio * (points.length - 1)), 0, points.length - 1);
    setHoverIndex(idx);
  };

  return (
    <div className="interactive-chart">
      <div className="chart-readout">
        <strong>Game {active.x} Mix:</strong>{" "}
        Build {active.build.toFixed(1)}% | Trade {active.trade.toFixed(1)}% | Dev{" "}
        {active.dev.toFixed(1)}% | Robber {active.robber.toFixed(1)}%
      </div>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <line
          x1={paddingLeft}
          y1={height - paddingY}
          x2={width - paddingRight}
          y2={height - paddingY}
          stroke="#505050"
        />
        <line
          x1={paddingLeft}
          y1={paddingY}
          x2={paddingLeft}
          y2={height - paddingY}
          stroke="#505050"
        />
        <text x={4} y={paddingY + 4} fill="#aaaaaa" fontSize="10">
          100%
        </text>
        <text x={8} y={height - paddingY + 4} fill="#aaaaaa" fontSize="10">
          0%
        </text>
        <path d={buildPath} fill="none" stroke="#4da3ff" strokeWidth={2.2} />
        <path d={tradePath} fill="none" stroke="#54d38b" strokeWidth={2.2} />
        <path d={devPath} fill="none" stroke="#c68cff" strokeWidth={2.2} />
        <path d={robberPath} fill="none" stroke="#f7c45a" strokeWidth={2.2} />
        {points.map((p, i) => (
          <circle
            key={`${p.x}-${i}`}
            cx={xAt(p.x)}
            cy={yAt(p.build + p.trade + p.dev + p.robber)}
            r={i === activeIndex ? 3.8 : 2}
            fill="#f7c45a"
            opacity={i === activeIndex ? 1 : 0.7}
            onClick={() => onPointClick?.(p)}
            style={{ cursor: onPointClick ? "pointer" : "default" }}
          />
        ))}
        <line x1={activeX} y1={paddingY} x2={activeX} y2={height - paddingY} stroke="#5d5d5d" strokeDasharray="3 3" />
      </svg>
      <div className="chart-legend">
        <span className="legend-item build">Build</span>
        <span className="legend-item trade">Trade</span>
        <span className="legend-item dev">Dev</span>
        <span className="legend-item robber">Robber</span>
      </div>
    </div>
  );
}

function CorrelationLineChart({
  titleLabel,
  points,
  color = "#64d8ff",
  onPointClick,
}: {
  titleLabel: string;
  points: CorrelationPoint[];
  color?: string;
  onPointClick?: (point: CorrelationPoint) => void;
}) {
  const width = 320;
  const height = 150;
  const paddingLeft = 34;
  const paddingRight = 10;
  const paddingY = 14;
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (points.length === 0) {
    return <div className="chart-empty">No data</div>;
  }

  const xMin = Math.min(...points.map((p) => p.x));
  const xMax = Math.max(...points.map((p) => p.x));
  const safeXMax = xMin === xMax ? xMax + 1 : xMax;
  const spanX = width - paddingLeft - paddingRight;
  const spanY = height - paddingY * 2;
  const xAt = (x: number) =>
    paddingLeft + (spanX * (x - xMin)) / (safeXMax - xMin);
  const yAt = (y: number) => height - paddingY - (spanY * y) / 100;

  const toPath = points
    .map((p, i) => {
      const x = xAt(p.x);
      const y = yAt(p.y);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  const activeIndex = hoverIndex ?? points.length - 1;
  const active = points[activeIndex];
  const activeX = xAt(active.x);
  const activeY = yAt(active.y);

  const handleMouseMove = (evt: MouseEvent<SVGSVGElement>) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const relativeX = clamp(
      evt.clientX - rect.left,
      paddingLeft,
      width - paddingRight
    );
    const ratio = (relativeX - paddingLeft) / spanX;
    const idx = clamp(Math.round(ratio * (points.length - 1)), 0, points.length - 1);
    setHoverIndex(idx);
  };

  return (
    <div className="interactive-chart">
      <div className="chart-readout">
        <strong>{titleLabel}:</strong> {active.label} | win rate{" "}
        {active.y.toFixed(1)}% | games {active.count}
      </div>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <line
          x1={paddingLeft}
          y1={height - paddingY}
          x2={width - paddingRight}
          y2={height - paddingY}
          stroke="#505050"
        />
        <line
          x1={paddingLeft}
          y1={paddingY}
          x2={paddingLeft}
          y2={height - paddingY}
          stroke="#505050"
        />
        <text x={8} y={paddingY + 4} fill="#aaaaaa" fontSize="10">
          100%
        </text>
        <text x={8} y={height - paddingY + 4} fill="#aaaaaa" fontSize="10">
          0%
        </text>
        <text x={paddingLeft - 4} y={height - 2} textAnchor="end" fill="#9d9d9d" fontSize="10">
          {xMin}
        </text>
        <text x={width - paddingRight + 2} y={height - 2} fill="#9d9d9d" fontSize="10">
          {xMax}
        </text>
        <path d={toPath} fill="none" stroke={color} strokeWidth={2.5} />
        {points.map((p, i) => (
          <circle
            key={`${p.x}-${i}`}
            cx={xAt(p.x)}
            cy={yAt(p.y)}
            r={i === activeIndex ? 4 : 2.5}
            fill={color}
            opacity={i === activeIndex ? 1 : 0.85}
            onClick={() => onPointClick?.(p)}
            style={{ cursor: onPointClick ? "pointer" : "default" }}
          />
        ))}
        <line
          x1={activeX}
          y1={paddingY}
          x2={activeX}
          y2={height - paddingY}
          stroke="#5d5d5d"
          strokeDasharray="3 3"
        />
        <circle cx={activeX} cy={activeY} r={4} fill={color} stroke="#111111" strokeWidth={1.5} />
      </svg>
    </div>
  );
}

export default function ReplayCatalogScreen() {
  const [rows, setRows] = useState<ReplayCatalogItem[] | null>(null);
  const [filters, setFilters] = useState<CatalogFilters>({
    won: "all",
    wentFirst: "all",
  });

  useEffect(() => {
    (async () => {
      const replays = await getReplayCatalog(500);
      setRows(replays);
    })();
  }, []);

  if (!rows) {
    return (
      <main className="replay-catalog-page loading">
        <GridLoader className="loader" color="#ffffff" size={50} />
      </main>
    );
  }

  const orderedRows = [...rows].reverse(); // game #1 is oldest
  const gameNumberById = new Map(orderedRows.map((row, i) => [row.game_id, i + 1]));
  const selectedGameId =
    filters.gameIndex !== undefined
      ? orderedRows[filters.gameIndex - 1]?.game_id
      : undefined;

  const filteredRows = rows.filter((row) => {
    if (selectedGameId && row.game_id !== selectedGameId) return false;
    if (filters.won === "yes" && row.won !== true) return false;
    if (filters.won === "no" && row.won !== false) return false;
    if (filters.wentFirst === "yes" && row.went_first !== true) return false;
    if (filters.wentFirst === "no" && row.went_first !== false) return false;
    if (
      filters.openingPipDiffExact !== undefined &&
      row.opening_pip_diff !== filters.openingPipDiffExact
    ) {
      return false;
    }
    if (filters.turnBucketStart !== undefined) {
      const size = filters.turnBucketSize ?? 50;
      const start = filters.turnBucketStart;
      if (!(row.turn_count >= start && row.turn_count < start + size)) {
        return false;
      }
    }
    return true;
  });

  const clearAllFilters = () =>
    setFilters({
      won: "all",
      wentFirst: "all",
    });

  const removeFilter = (
    key: "gameIndex" | "openingPipDiffExact" | "turnBucketStart"
  ) => {
    setFilters((prev) => {
      const next: CatalogFilters = { ...prev };
      if (key === "gameIndex") delete next.gameIndex;
      if (key === "openingPipDiffExact") delete next.openingPipDiffExact;
      if (key === "turnBucketStart") {
        delete next.turnBucketStart;
        delete next.turnBucketSize;
      }
      return next;
    });
  };

  const total = rows.length;
  const wins = rows.filter((row) => row.won).length;
  const losses = rows.filter((row) => row.won === false).length;
  const firstRows = rows.filter((row) => row.went_first === true);
  const secondRows = rows.filter((row) => row.went_first === false);
  const firstWins = firstRows.filter((row) => row.won).length;
  const secondWins = secondRows.filter((row) => row.won).length;

  const winRows = rows.filter((row) => row.won === true);
  const lossRows = rows.filter((row) => row.won === false);
  const avgTurnsWin = average(winRows.map((row) => row.turn_count));
  const avgTurnsLoss = average(lossRows.map((row) => row.turn_count));
  const avgVpWin = average(
    winRows.map((row) => row.us_final_vp ?? 0).filter((v) => v > 0)
  );
  const avgVpLoss = average(
    lossRows.map((row) => row.us_final_vp ?? 0).filter((v) => v > 0)
  );
  const avgOpeningPips = averageNullable(rows.map((row) => row.us_opening_pip_score));
  const avgOpeningPipDiff = averageNullable(rows.map((row) => row.opening_pip_diff));
  const avgFirstCityTurn = averageNullable(rows.map((row) => row.us_first_city_turn));
  const positivePipDiffRows = rows.filter((row) => (row.opening_pip_diff ?? 0) > 0);
  const neutralPipDiffRows = rows.filter((row) => (row.opening_pip_diff ?? 0) === 0);
  const negativePipDiffRows = rows.filter((row) => (row.opening_pip_diff ?? 0) < 0);
  const positivePipDiffWins = positivePipDiffRows.filter((row) => row.won).length;
  const neutralPipDiffWins = neutralPipDiffRows.filter((row) => row.won).length;
  const negativePipDiffWins = negativePipDiffRows.filter((row) => row.won).length;

  const devBehavior = splitBehaviorWinRate(rows, (row) => row.us_buy_dev);
  const tradeBehavior = splitBehaviorWinRate(rows, (row) => row.us_maritime_trades);
  const knightBehavior = splitBehaviorWinRate(rows, (row) => row.us_play_knight);

  const cumulativeWinRate = cumulativeWinRatePoints(rows);
  const rollingWinRate = rollingWinRatePoints(rows, 10);
  const turnsSeries = turnCountPoints(rows);
  const vpDiffSeries = vpDiffPoints(rows);
  const openingPipDiffSeries = openingPipDiffPoints(rows);
  const openingPipWinRateSeries = pipDiffWinRateCorrelation(rows);
  const turnBucketWinRateSeries = turnBucketWinRate(rows, 50);
  const firstCitySeries = firstCityPoints(rows);
  const mixSeries = actionMixPoints(rows);

  const avgDevBuys = average(rows.map((row) => row.us_buy_dev));
  const avgTrades = average(rows.map((row) => row.us_maritime_trades));
  const avgKnights = average(rows.map((row) => row.us_play_knight));
  const avgCities = average(rows.map((row) => row.us_build_city));
  const avgSettlements = average(rows.map((row) => row.us_build_settlement));
  const totalDevBuys = sum(rows.map((row) => row.us_buy_dev));
  const totalTrades = sum(rows.map((row) => row.us_maritime_trades));

  const headlineStats: StatRow[] = [
    { label: "Games", value: `${total}` },
    { label: "Wins / Losses", value: `${wins} / ${losses}` },
    { label: "Win Rate", value: `${pct(wins, total).toFixed(1)}%` },
    {
      label: "Win Rate (Went First)",
      value: `${pct(firstWins, firstRows.length).toFixed(1)}%`,
    },
    {
      label: "Win Rate (Went Second)",
      value: `${pct(secondWins, secondRows.length).toFixed(1)}%`,
    },
    { label: "Avg Turns (Wins)", value: `${avgTurnsWin.toFixed(1)}` },
    { label: "Avg Turns (Losses)", value: `${avgTurnsLoss.toFixed(1)}` },
    { label: "Avg Final VP (Wins)", value: `${avgVpWin.toFixed(2)}` },
    { label: "Avg Final VP (Losses)", value: `${avgVpLoss.toFixed(2)}` },
    { label: "Avg Opening Pip Score", value: `${avgOpeningPips.toFixed(2)}` },
    { label: "Avg Opening Pip Differential", value: `${avgOpeningPipDiff.toFixed(2)}` },
    { label: "Avg First City Turn", value: `${avgFirstCityTurn.toFixed(1)}` },
    { label: "Avg Dev Buys / Game", value: `${avgDevBuys.toFixed(2)}` },
    { label: "Avg Maritime Trades / Game", value: `${avgTrades.toFixed(2)}` },
    { label: "Avg Knight Plays / Game", value: `${avgKnights.toFixed(2)}` },
    { label: "Avg Cities Built / Game", value: `${avgCities.toFixed(2)}` },
    { label: "Avg Settlements Built / Game", value: `${avgSettlements.toFixed(2)}` },
    { label: "Total Dev Buys", value: `${totalDevBuys}` },
    { label: "Total Maritime Trades", value: `${totalTrades}` },
  ];

  return (
    <main className="replay-catalog-page">
      <h1 className="logo">Replay Catalog</h1>
      <div className="catalog-card foreground">
        <div className="catalog-header">
          <div>
            Showing {filteredRows.length} / {rows.length} replay(s)
          </div>
          <Button component={Link} variant="contained" to="/">
            Home
          </Button>
        </div>

        <div className="catalog-filters">
          <label>
            Won:
            <select
              value={filters.won}
              onChange={(e) =>
                setFilters((prev) => ({
                  ...prev,
                  won: e.target.value as CatalogFilters["won"],
                }))
              }
            >
              <option value="all">All</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </label>

          <label>
            Went First:
            <select
              value={filters.wentFirst}
              onChange={(e) =>
                setFilters((prev) => ({
                  ...prev,
                  wentFirst: e.target.value as CatalogFilters["wentFirst"],
                }))
              }
            >
              <option value="all">All</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </label>

          <Button size="small" variant="outlined" onClick={clearAllFilters}>
            Clear All Filters
          </Button>
        </div>

        <div className="active-filters">
          {filters.gameIndex !== undefined && (
            <button type="button" className="filter-chip" onClick={() => removeFilter("gameIndex")}>
              Game #{filters.gameIndex} x
            </button>
          )}
          {filters.openingPipDiffExact !== undefined && (
            <button type="button" className="filter-chip" onClick={() => removeFilter("openingPipDiffExact")}>
              Pip diff = {filters.openingPipDiffExact} x
            </button>
          )}
          {filters.turnBucketStart !== undefined && (
            <button type="button" className="filter-chip" onClick={() => removeFilter("turnBucketStart")}>
              Turns {filters.turnBucketStart}-{filters.turnBucketStart + (filters.turnBucketSize ?? 50) - 1} x
            </button>
          )}
        </div>

        <div className="catalog-body">
          <div className="catalog-table">
            <table>
              <thead>
                <tr>
                  <th>Replay ID</th>
                  <th>Folder</th>
                  <th>Uploaded</th>
                  <th>Game #</th>
                  <th>Turn #</th>
                  <th>State #</th>
                  <th>Went First</th>
                  <th>Won</th>
                  <th>Open</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row) => (
                  <tr key={row.game_id}>
                    <td className="mono">{row.game_id}</td>
                    <td className="mono">{row.replay_source_folder ?? "N/A"}</td>
                    <td>{formatImportedAt(row.imported_at_utc)}</td>
                    <td>{gameNumberById.get(row.game_id) ?? "N/A"}</td>
                    <td>{row.turn_count}</td>
                    <td>{row.state_index}</td>
                    <td>{formatWentFirst(row.went_first)}</td>
                    <td>{formatWon(row.won, row.winner)}</td>
                    <td>
                      <Button
                        component={Link}
                        size="small"
                        variant="outlined"
                        to={`/replays/${row.game_id}`}
                      >
                        View
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <aside className="stats-panel">
            <h3>Optimization Stats</h3>
            <div className="stats-block">
              {headlineStats.map((stat) => (
                <div className="stat-row" key={stat.label}>
                  <span>{stat.label}</span>
                  <strong>{stat.value}</strong>
                </div>
              ))}
            </div>

            <div className="stats-block">
              <h4>Cumulative Win Rate Over Game #</h4>
              <InteractiveLineChart
                points={cumulativeWinRate}
                yLabel="Cumulative Win Rate"
                yMin={0}
                yMax={100}
                color="#4da3ff"
                formatY={(v) => `${v.toFixed(1)}%`}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>Rolling Win Rate (10 games)</h4>
              <InteractiveLineChart
                points={rollingWinRate}
                yLabel="Rolling Win Rate"
                yMin={0}
                yMax={100}
                color="#54d38b"
                formatY={(v) => `${v.toFixed(1)}%`}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>Turns Per Game</h4>
              <InteractiveLineChart
                points={turnsSeries}
                yLabel="Turns"
                color="#f7c45a"
                formatY={(v) => `${v.toFixed(0)}`}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>VP Differential (US - Opp)</h4>
              <InteractiveLineChart
                points={vpDiffSeries}
                yLabel="VP Differential"
                color="#c68cff"
                formatY={(v) => `${v.toFixed(2)}`}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>Opening Pip Differential Over Game #</h4>
              <InteractiveLineChart
                points={openingPipDiffSeries}
                yLabel="Opening Pip Differential"
                color="#64d8ff"
                formatY={(v) => `${v.toFixed(0)}`}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>Pip Differential vs Win Rate</h4>
              <CorrelationLineChart
                titleLabel="Pip Differential"
                points={openingPipWinRateSeries}
                color="#64d8ff"
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    openingPipDiffExact: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>First City Timing</h4>
              <InteractiveLineChart
                points={firstCitySeries}
                yLabel="First City Turn"
                color="#ff9f6e"
                formatY={(v) => `${v.toFixed(0)}`}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>First vs Second Win Rate</h4>
              <SplitBarChart
                firstWinRate={pct(firstWins, firstRows.length)}
                secondWinRate={pct(secondWins, secondRows.length)}
                firstCount={firstRows.length}
                secondCount={secondRows.length}
              />
            </div>

            <div className="stats-block">
              <h4>Action Mix Over Game # (US)</h4>
              <ActionMixChart
                points={mixSeries}
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    gameIndex: point.x,
                  }))
                }
              />
            </div>

            <div className="stats-block">
              <h4>Behavior vs Win Rate</h4>
              <div className="stat-row">
                <span>High Dev Buys (&gt;= {devBehavior.median})</span>
                <strong>{devBehavior.highWinRate.toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>Low Dev Buys (&lt; {devBehavior.median})</span>
                <strong>{devBehavior.lowWinRate.toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>High Maritime Trades (&gt;= {tradeBehavior.median})</span>
                <strong>{tradeBehavior.highWinRate.toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>Low Maritime Trades (&lt; {tradeBehavior.median})</span>
                <strong>{tradeBehavior.lowWinRate.toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>High Knight Plays (&gt;= {knightBehavior.median})</span>
                <strong>{knightBehavior.highWinRate.toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>Low Knight Plays (&lt; {knightBehavior.median})</span>
                <strong>{knightBehavior.lowWinRate.toFixed(1)}%</strong>
              </div>
            </div>

            <div className="stats-block">
              <h4>Turns per Game Bucket vs Win Rate (50-turn bins)</h4>
              <CorrelationLineChart
                titleLabel="Turns Bucket"
                points={turnBucketWinRateSeries}
                color="#f7c45a"
                onPointClick={(point) =>
                  setFilters((prev) => ({
                    ...prev,
                    turnBucketStart: point.x,
                    turnBucketSize: 50,
                  }))
                }
              />
              <div className="stat-row">
                <span>Positive Pip Diff Win Rate</span>
                <strong>{pct(positivePipDiffWins, positivePipDiffRows.length).toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>Neutral Pip Diff Win Rate</span>
                <strong>{pct(neutralPipDiffWins, neutralPipDiffRows.length).toFixed(1)}%</strong>
              </div>
              <div className="stat-row">
                <span>Negative Pip Diff Win Rate</span>
                <strong>{pct(negativePipDiffWins, negativePipDiffRows.length).toFixed(1)}%</strong>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
