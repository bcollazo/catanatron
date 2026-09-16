import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import {
  Cpu,
  User,
  Route,
  Shield,
  Activity,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  SkipBack,
  SkipForward,
} from "lucide-react";
import {
  type GameState,
  title,
  stat,
  playerColors,
  recordLabel,
} from "./model";
import { ResourceHand } from "./Resources";
import { api } from "./api";
export function Players({ game }: { game: GameState }) {
  return (
    <section className="players-panel">
      <div className="panel-title">
        <span className="eyebrow">PLAYERS</span>
      </div>
      {game.colors.map((color) => (
        <article
          className={`player-panel ${game.current_color === color ? "current" : ""}`}
          style={{ "--player": playerColors[color] } as React.CSSProperties}
          key={color}
        >
          <div className="player-head">
            <div className="avatar">
              {game.bot_colors.includes(color) ? (
                <Cpu size={19} />
              ) : (
                <User size={19} />
              )}
            </div>
            <div>
              <strong>{title(color)}</strong>
              <small>
                {game.bot_colors.includes(color)
                  ? "AI opponent"
                  : "Human player"}
              </small>
            </div>
            <motion.div
              className="points"
              key={stat(game, color, "ACTUAL_VICTORY_POINTS")}
              initial={{ scale: 1.2 }}
              animate={{ scale: 1 }}
            >
              {stat(game, color, "ACTUAL_VICTORY_POINTS")}
              <small>VP</small>
            </motion.div>
          </div>
          <ResourceHand game={game} color={color} compact />
          <div className="player-stats">
            <span title="Longest road">
              <Route size={14} />
              {game.longest_roads_by_player?.[color] ??
                stat(game, color, "LONGEST_ROAD_LENGTH")}
              {stat(game, color, "HAS_ROAD") ? " ★" : ""}
            </span>
            <span title="Knights played">
              <Shield size={14} />
              {stat(game, color, "PLAYED_KNIGHT")}
              {stat(game, color, "HAS_ARMY") ? " ★" : ""}
            </span>
          </div>
        </article>
      ))}
    </section>
  );
}
export function ActivityLog({ game }: { game: GameState }) {
  const [filter, setFilter] = useState("ALL");
  const records = game.action_records
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => filter === "ALL" || r[0][0] === filter)
    .reverse();
  const [limit, setLimit] = useState(50);
  return (
    <section className="activity-panel">
      <div className="panel-title">
        <span className="eyebrow">ACTIVITY</span>
        <select
          aria-label="Filter activity by player"
          value={filter}
          onChange={(e) => {
            setFilter(e.target.value);
            setLimit(50);
          }}
        >
          <option value="ALL">All players</option>
          {game.colors.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      </div>
      <div className="activity-list">
        {records.length === 0 ? (
          <p className="muted">The first move is yours.</p>
        ) : (
          records.slice(0, limit).map(({ r, i }) => (
            <motion.div
              className="log-entry"
              key={i}
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <span className="log-index">
                {String(i + 1).padStart(3, "0")}
              </span>
              <span
                className="log-dot"
                style={{ background: playerColors[r[0][0]] }}
              />
              <div>
                <small style={{ color: playerColors[r[0][0]] }}>
                  {title(r[0][0])}
                </small>
                <p>{recordLabel(r)}</p>
              </div>
            </motion.div>
          ))
        )}
        {records.length > limit && (
          <button onClick={() => setLimit((l) => l + 100)}>
            Show more moves
          </button>
        )}
      </div>
    </section>
  );
}
export function Analysis({ id, game }: { id?: string; game: GameState }) {
  const [results, setResults] = useState<Record<string, number> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  useEffect(() => {
    setResults(null);
    setError("");
    setBusy(false);
    return () => request.current?.abort();
  }, [id, game.state_index]);
  const analyze = () => {
    if (busy || !id) return;
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setError("");
    setBusy(true);
    api
      .analyze(id, game.state_index, controller.signal)
      .then((r) => {
        if (!controller.signal.aborted) {
          if (!r.success) throw new Error(r.error || "Analysis failed");
          setResults(r.probabilities);
        }
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setBusy(false);
        }
      });
  };
  return (
    <section className="analysis-panel">
      <div className="panel-title">
        <span className="eyebrow">POSITION ANALYSIS</span>
        <Activity size={16} />
      </div>
      <p className="muted">Estimate this position with 100 simulations.</p>
      <button
        className="analysis-button"
        disabled={!id || busy || !!game.winning_color}
        onClick={analyze}
      >
        <Cpu size={17} />
        {busy ? "Simulating…" : "Analyze position"}
      </button>
      {!id && (
        <small className="muted">
          Analysis requires a game saved on the server.
        </small>
      )}
      {error && <p role="alert">{error}</p>}
      {results && (
        <>
          <div className="probabilities">
            {Object.entries(results).map(([c, p]) => (
              <div key={c}>
                <span>{title(c)}</span>
                <div>
                  <motion.i
                    initial={{ width: 0 }}
                    animate={{ width: `${p}%` }}
                    style={{
                      background: playerColors[c as keyof typeof playerColors],
                    }}
                  />
                </div>
                <strong>{p}%</strong>
              </div>
            ))}
          </div>
          <small className="muted">
            Approximate: the engine splits non-current-player wins equally among
            opponents.
          </small>
        </>
      )}
    </section>
  );
}
export function ReplayControls({
  index,
  max,
  setIndex,
  playing,
  setPlaying,
  speed,
  setSpeed,
  busy,
}: {
  index: number;
  max: number;
  setIndex: (n: number) => void;
  playing: boolean;
  setPlaying: (b: boolean) => void;
  speed: number;
  setSpeed: (n: number) => void;
  busy: boolean;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        (event.key !== "ArrowLeft" && event.key !== "ArrowRight") ||
        document.querySelector("dialog[open]")
      )
        return;
      const target = event.target;
      if (
        target instanceof Element &&
        target.closest(
          'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="slider"], [role="spinbutton"]',
        )
      )
        return;
      event.preventDefault();
      setPlaying(false);
      const next = Math.max(
        0,
        Math.min(max, index + (event.key === "ArrowRight" ? 1 : -1)),
      );
      if (next !== index) setIndex(next);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [index, max, setIndex, setPlaying]);
  return (
    <section className="replay-controls">
      <div className="panel-title">
        <span className="eyebrow">REPLAY TIMELINE</span>
        <span>
          {index} / {max}
        </span>
      </div>
      <input
        type="range"
        aria-label="Replay position"
        min={0}
        max={max}
        value={index}
        disabled={!max}
        onChange={(e) => {
          setPlaying(false);
          setIndex(Number(e.target.value));
        }}
      />
      <div className="replay-buttons">
        <button
          aria-label="First move"
          disabled={!index}
          onClick={() => {
            setPlaying(false);
            setIndex(0);
          }}
        >
          <SkipBack size={17} />
        </button>
        <button
          aria-label="Previous move"
          aria-keyshortcuts="ArrowLeft"
          title="Previous move (←)"
          disabled={!index}
          onClick={() => {
            setPlaying(false);
            setIndex(index - 1);
          }}
        >
          <ChevronLeft size={18} />
        </button>
        <button
          className="primary"
          aria-label={playing ? "Pause replay" : "Play replay"}
          disabled={!max || busy}
          onClick={() => {
            if (index === max) setIndex(0);
            setPlaying(!playing);
          }}
        >
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button
          aria-label="Next move"
          aria-keyshortcuts="ArrowRight"
          title="Next move (→)"
          disabled={index >= max}
          onClick={() => {
            setPlaying(false);
            setIndex(index + 1);
          }}
        >
          <ChevronRight size={18} />
        </button>
        <button
          aria-label="Last move"
          disabled={index >= max}
          onClick={() => {
            setPlaying(false);
            setIndex(max);
          }}
        >
          <SkipForward size={17} />
        </button>
        <select
          aria-label="Replay speed"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        >
          <option value={1600}>0.5×</option>
          <option value={800}>1×</option>
          <option value={400}>2×</option>
          <option value={160}>5×</option>
        </select>
        <label className="jump-label">
          Go to
          <input
            aria-label="Go to move"
            type="number"
            min={0}
            max={max}
            value={index}
            onChange={(e) => {
              setPlaying(false);
              setIndex(
                Math.max(0, Math.min(max, Math.trunc(Number(e.target.value)))),
              );
            }}
          />
        </label>
      </div>
    </section>
  );
}
