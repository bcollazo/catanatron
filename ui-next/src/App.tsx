import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, MotionConfig } from "motion/react";
import {
  Hexagon,
  ArrowUpRight,
  ArrowRight,
  Upload,
  Plus,
  X,
  Cpu,
  Play,
  Pause,
  ChevronLeft,
  RotateCcw,
  Download,
  Trophy,
  Dices,
  BookOpen,
  GitFork,
} from "lucide-react";
import { api } from "./api";
import { Board } from "./Board";
import { GameSettingsButton } from "./GameSettings";
import { Legend, ResourceHand } from "./Resources";
import { ActionDock, ChoiceDialog } from "./ActionDock";
import { ActivityLog, Analysis, Players, ReplayControls } from "./Panels";
import { type Session, useGame } from "./useGame";
import {
  type Action,
  type GameState,
  parseReplay,
  combineSnapshots,
  recordLabel,
  title,
  playerColors,
} from "./model";

function readRoute(): Session | null {
  const replay = location.pathname.match(/^\/replays\/([^/]+)$/);
  if (replay) return { id: decodeURIComponent(replay[1]), replay: true };
  const game = location.pathname.match(
    /^\/games\/([^/]+)(?:\/states\/(\d+))?$/,
  );
  return game
    ? {
        id: decodeURIComponent(game[1]),
        replay: !!game[2],
        index: game[2] ? Number(game[2]) : undefined,
      }
    : null;
}
function Brand() {
  return (
    <div className="brand">
      <div className="brand-icon">
        <Hexagon size={25} />
        <span />
      </div>
      <span>catanatron</span>
    </div>
  );
}
function HomeScreen({ open }: { open: (s: Session) => void }) {
  const [entry, setEntry] = useState("create");
  const [players, setPlayers] = useState(["HUMAN", "CATANATRON"]);
  const [map, setMap] = useState("BASE");
  const [points, setPoints] = useState(15);
  const [discard, setDiscard] = useState(9);
  const [friendly, setFriendly] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [id, setId] = useState("");
  const [sample, setSample] = useState<GameState | null>(null);
  const file = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const c = new AbortController();
    fetch("/example-preview.json", { signal: c.signal })
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.text();
      })
      .then((t) => setSample(parseReplay(t)[0]))
      .catch(() => {});
    return () => c.abort();
  }, []);
  const openExample = async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/example-game.json");
      if (!response.ok)
        throw new Error("The example replay could not be loaded.");
      open({
        local: parseReplay(await response.text()),
        replay: true,
        label: "Example · recorded game",
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const create = async () => {
    setBusy(true);
    setError("");
    try {
      const { game_id } = await api.create({
        players,
        map_template: map,
        vps_to_win: points,
        discard_limit: discard,
        friendly_robber: friendly,
      });
      open({ id: game_id, replay: false });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const importFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setError("");
    setBusy(true);
    try {
      if (Array.from(files).reduce((s, f) => s + f.size, 0) > 50 * 1024 * 1024)
        throw new Error("Choose a replay smaller than 50 MB.");
      const states = combineSnapshots(
        (
          await Promise.all(
            Array.from(files).map(async (f) => parseReplay(await f.text())),
          )
        ).flat(),
      );
      open({
        local: states,
        replay: true,
        label: files.length === 1 ? files[0].name : `${files.length} snapshots`,
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (file.current) file.current.value = "";
    }
  };
  return (
    <div className="home-screen">
      <header className="topbar">
        <Brand />
        <nav className="home-links" aria-label="Project links">
          <a href="https://docs.catanatron.com/">Docs</a>
          <a href="https://github.com/bcollazo/catanatron">
            <GitFork size={17} /> GitHub
          </a>
        </nav>
      </header>
      <main className="setup-layout">
        <section className="setup-panel">
          <span className="eyebrow accent">OPEN SOURCE · CATAN AI</span>
          <h1>
            Play Catan.
            <br />
            <span>Build better bots.</span>
          </h1>
          <p className="intro">
            Challenge a bot or inspect a recorded game. Explore Catan strategy
            with a fast, open-source simulator.
          </p>
          <div
            className="entry-tabs"
            role="tablist"
            aria-label="Choose how to begin"
          >
            <button
              id="create-tab"
              role="tab"
              aria-selected={entry === "create"}
              aria-controls="create-panel"
              onClick={() => setEntry("create")}
            >
              <Plus size={17} /> Create a game
            </button>
            <button
              id="inspect-tab"
              role="tab"
              aria-selected={entry === "inspect"}
              aria-controls="inspect-panel"
              onClick={() => setEntry("inspect")}
            >
              <BookOpen size={17} /> Inspect a game
            </button>
          </div>
          <div
            className="entry-panel"
            id="create-panel"
            role="tabpanel"
            aria-labelledby="create-tab"
            hidden={entry !== "create"}
          >
            <div className="setup-section">
              <div className="section-label">
                <span>01</span>
                <h2>The board</h2>
              </div>
              <div className="map-options">
                {["BASE", "MINI", "TOURNAMENT"].map((m) => (
                  <button
                    key={m}
                    className={map === m ? "selected" : ""}
                    aria-pressed={map === m}
                    onClick={() => setMap(m)}
                  >
                    <Hexagon size={22} />
                    <strong>{title(m)}</strong>
                    <small>
                      {m === "BASE"
                        ? "Classic island"
                        : m === "MINI"
                          ? "Quick encounters"
                          : "Balanced terrain"}
                    </small>
                  </button>
                ))}
              </div>
              <div className="rule-inputs">
                <label>
                  Points to win
                  <input
                    type="number"
                    min="3"
                    max="20"
                    value={points}
                    onChange={(e) => setPoints(Number(e.target.value))}
                  />
                </label>
                <label>
                  Discard limit
                  <input
                    type="number"
                    min="5"
                    max="20"
                    value={discard}
                    onChange={(e) => setDiscard(Number(e.target.value))}
                  />
                </label>
              </div>
              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={friendly}
                  onChange={(e) => setFriendly(e.target.checked)}
                />
                <span>
                  Friendly robber
                  <small>Protect opponents with 2 victory points</small>
                </span>
              </label>
            </div>
            <div className="setup-section">
              <div className="section-label">
                <span>02</span>
                <h2>The players</h2>
                <span className="muted">{players.length} / 4</span>
              </div>
              <div className="setup-players">
                {players.map((p, i) => (
                  <div className="setup-player" key={i}>
                    <span
                      className="seat"
                      style={{ color: Object.values(playerColors)[i] }}
                    >
                      0{i + 1}
                    </span>
                    <select
                      aria-label={`Player ${i + 1}`}
                      value={p}
                      onChange={(e) =>
                        setPlayers((ps) =>
                          ps.map((v, j) => (j === i ? e.target.value : v)),
                        )
                      }
                    >
                      {[
                        ["HUMAN", "Human · you"],
                        ["CATANATRON", "Catanatron · alpha-beta"],
                        ["WEIGHTED_RANDOM", "Weighted random"],
                        ["RANDOM", "Random"],
                      ].map(([v, l]) => (
                        <option
                          key={v}
                          value={v}
                          disabled={
                            v === "HUMAN" &&
                            p !== "HUMAN" &&
                            players.includes("HUMAN")
                          }
                        >
                          {l}
                        </option>
                      ))}
                    </select>
                    <button
                      aria-label={`Remove player ${i + 1}`}
                      disabled={players.length <= 2}
                      onClick={() =>
                        setPlayers((ps) => ps.filter((_, j) => j !== i))
                      }
                    >
                      <X size={17} />
                    </button>
                  </div>
                ))}
              </div>
              <button
                className="add-player"
                disabled={players.length >= 4}
                onClick={() => setPlayers((p) => [...p, "WEIGHTED_RANDOM"])}
              >
                <Plus size={16} />
                Add opponent
              </button>
            </div>
            <button
              className="primary start-button"
              disabled={
                busy ||
                points < 3 ||
                points > 20 ||
                discard < 5 ||
                discard > 20 ||
                !Number.isInteger(points) ||
                !Number.isInteger(discard)
              }
              onClick={() => void create()}
            >
              {busy ? "Connecting…" : "Start game"}
              <ArrowRight size={20} />
            </button>
            <p className="setup-footnote">2–4 players · one human per game</p>
          </div>
          <div
            className="entry-panel"
            id="inspect-panel"
            role="tabpanel"
            aria-labelledby="inspect-tab"
            hidden={entry !== "inspect"}
          >
            <div className="inspect-card">
              <div className="inspect-heading">
                <BookOpen size={22} />
                <div>
                  <h2>Understand how your bot plays.</h2>
                  <p>
                    Inspect a CLI game, replay a match, and find the next
                    improvement.
                  </p>
                </div>
              </div>
              <div className="import-actions">
                <button disabled={busy} onClick={() => file.current?.click()}>
                  <Upload size={18} /> Import JSON
                  <ArrowUpRight size={16} />
                </button>
                <button disabled={busy} onClick={() => void openExample()}>
                  <Play size={16} /> Explore a replay
                </button>
              </div>
              <input
                ref={file}
                type="file"
                accept=".json,application/json"
                multiple
                hidden
                onChange={(e) => void importFiles(e.target.files)}
              />
              <form
                className="open-game"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (id.trim()) open({ id: id.trim(), replay: true });
                }}
              >
                <input
                  aria-label="Saved game ID"
                  placeholder="Paste a saved game ID"
                  value={id}
                  onChange={(e) => setId(e.target.value)}
                />
                <button type="submit" disabled={!id.trim()}>
                  Inspect <ArrowRight size={16} />
                </button>
              </form>
              <details>
                <summary>Using CLI games</summary>
                <p>
                  Export with <code>--output games --output-format json</code>{" "}
                  for a final board and action log. For a full timeline, save to
                  the same database with <code>--step-db</code>, then paste the
                  game ID. You can also import an array of snapshots.
                </p>
              </details>
            </div>
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
        </section>
        <section className="preview-panel">
          <div className="preview-board">
            {sample ? (
              <Board game={sample} animated={false} />
            ) : (
              <div className="preview-placeholder">
                <Hexagon size={110} strokeWidth={0.5} />
                <span>CATANATRON</span>
              </div>
            )}
          </div>
          <Legend />
        </section>
      </main>
      <footer className="home-footer">
        <span>
          CATANATRON <span className="accent">/</span> OPEN SOURCE
        </span>
        <a
          className="contribute-inline"
          href="https://github.com/bcollazo/catanatron"
        >
          <GitFork size={16} /> Help make Catanatron stronger. Contribute on
          GitHub <ArrowUpRight size={15} />
        </a>
      </footer>
    </div>
  );
}

function GameScreen({
  session,
  open,
  home,
}: {
  session: Session;
  open: (s: Session) => void;
  home: () => void;
}) {
  const g = useGame(session);
  const [mode, setMode] = useState("");
  const [choices, setChoices] = useState<Action[] | null>(null);
  const [panel, setPanel] = useState("Board");
  const [sidebarTab, setSidebarTab] = useState("Players");
  const detailTab = panel === "Board" ? sidebarTab : panel;
  const game = g.game;
  useEffect(() => {
    setMode("");
    setChoices(null);
  }, [game?.state_index]);
  const act = (a: Action) => {
    setChoices(null);
    void g.act(a);
  };
  const exportGame = () => {
    if (!game) return;
    const blob = new Blob([JSON.stringify(session.local ?? game, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `catanatron-${session.id || "replay"}-${game.state_index}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const human = game?.colors.find((c) => !game.bot_colors.includes(c));
  const canAct =
    game &&
    !session.replay &&
    !g.busy &&
    !g.error &&
    !game.winning_color &&
    !game.bot_colors.includes(game.current_color);
  const boardActions = canAct
    ? game.current_playable_actions.filter(
        (a) =>
          a[1] === mode ||
          (game.is_initial_build_phase && a[1].startsWith("BUILD_")) ||
          (game.current_prompt === "MOVE_ROBBER" && a[1] === "MOVE_ROBBER") ||
          (game.current_playable_actions.every((a) => a[1] === "BUILD_ROAD") &&
            a[1] === "BUILD_ROAD"),
      )
    : [];
  const last = game?.action_records.at(-1);
  const announcement = (
    <AnimatePresence mode="wait">
      {last && (
        <motion.div
          key={game?.state_index}
          className={`move-event event-${last[0][1].toLowerCase()}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
        >
          <span style={{ color: playerColors[last[0][0]] }}>
            {title(last[0][0])}
          </span>
          {last[0][1] === "ROLL" ? (
            <motion.span
              className="dice-pair"
              initial={{ rotate: -30, scale: 0.6 }}
              animate={{ rotate: 0, scale: 1 }}
            >
              <Dices size={20} />
            </motion.span>
          ) : last[0][1].startsWith("PLAY_") ? (
            <Cpu size={17} />
          ) : null}
          <span>{recordLabel(last)}</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
  return (
    <div className="game-screen">
      <header className="topbar">
        <button
          className="back-button"
          aria-label="Back to setup"
          onClick={home}
        >
          <ChevronLeft size={20} />
        </button>
        <Brand />
        <div className="game-meta">
          <span className="live-badge">
            {session.replay ? "REPLAY" : "LIVE GAME"}
          </span>
          <span className="mono">{session.id?.slice(0, 8) || "LOCAL"}</span>
        </div>
        <div className="header-actions">
          {game && <GameSettingsButton game={game} />}
          {session.id && (
            <button
              onClick={() =>
                open({
                  ...session,
                  replay: !session.replay,
                  index: session.replay ? undefined : 0,
                })
              }
            >
              {session.replay ? <Play size={16} /> : <RotateCcw size={16} />}
              <span>{session.replay ? "Resume game" : "Replay"}</span>
            </button>
          )}
          <button
            disabled={!game}
            aria-label="Download game JSON"
            onClick={exportGame}
          >
            <Download size={17} />
          </button>
        </div>
      </header>
      {g.error && (
        <div className="error" role="alert">
          {g.error}
          <button onClick={() => void g.refresh()}>Reload game</button>
        </div>
      )}
      {!game ? (
        <div className="loading-screen">
          <Cpu className="loading-icon" size={40} />
          <h2>
            {g.error ? "Unable to open this game" : "Loading the island…"}
          </h2>
          <button onClick={home}>Back to setup</button>
        </div>
      ) : (
        <>
          <nav className="mobile-tabs" aria-label="Game panels">
            {["Board", "Players", "Activity", "Analysis"].map((p) => (
              <button
                key={p}
                aria-pressed={panel === p}
                onClick={() => setPanel(p)}
              >
                {p}
              </button>
            ))}
          </nav>
          <main className={`game-layout show-${panel.toLowerCase()}`}>
            <section className="center-column">
              <div className="board-stage">
                <Board
                  game={game}
                  actions={boardActions}
                  onAction={act}
                  onChoices={setChoices}
                />

                {game.winning_color && (
                  <motion.div
                    className="victory-mark"
                    initial={{ scale: 0, rotate: -30 }}
                    animate={{ scale: 1, rotate: 0 }}
                  >
                    <Trophy size={26} />
                  </motion.div>
                )}
              </div>
              <Legend />
              {session.replay ? (
                <>
                  <div className="replay-announcement" role="status">
                    {announcement}
                  </div>
                  <ReplayControls {...g} />
                  {session.local?.length === 1 && (
                    <p className="snapshot-note">
                      Final snapshot · the full action log is available, but
                      this export has no intermediate board states.
                    </p>
                  )}
                </>
              ) : (
                <>
                  <div className="hand-heading">
                    <span className="eyebrow">
                      {human ? "YOUR HAND" : `${game.current_color}'S HAND`}
                    </span>
                  </div>
                  <ResourceHand
                    game={game}
                    color={human || game.current_color}
                  />
                  <ActionDock
                    game={game}
                    busy={g.busy || !!g.error}
                    mode={mode}
                    setMode={setMode}
                    onAction={act}
                    onChoices={setChoices}
                    botControls={
                      game.bot_colors.includes(game.current_color) &&
                      !game.winning_color && (
                        <div className="bot-controls">
                          <button onClick={() => g.setPaused(!g.paused)}>
                            {g.paused ? (
                              <Play size={16} />
                            ) : (
                              <Pause size={16} />
                            )}{" "}
                            {g.paused ? "Resume bots" : "Pause bots"}
                          </button>
                          <select
                            aria-label="Bot move speed"
                            value={g.speed}
                            onChange={(e) => g.setSpeed(Number(e.target.value))}
                          >
                            <option value={1600}>Relaxed</option>
                            <option value={800}>Normal</option>
                            <option value={300}>Fast</option>
                          </select>
                        </div>
                      )
                    }
                  />
                </>
              )}
            </section>
            <aside className="game-sidebar" aria-label="Game details">
              <nav className="sidebar-tabs" aria-label="Sidebar panels">
                {["Players", "Activity", "Analysis"].map((tab) => (
                  <button
                    key={tab}
                    aria-pressed={detailTab === tab}
                    onClick={() => {
                      setSidebarTab(tab);
                      setPanel("Board");
                    }}
                  >
                    {tab}
                  </button>
                ))}
              </nav>
              <div hidden={detailTab !== "Players"}>
                <Players game={game} />
              </div>
              <div className="analysis-wrap" hidden={detailTab !== "Analysis"}>
                <Analysis id={session.id} game={game} />
              </div>
              <div className="activity-wrap" hidden={detailTab !== "Activity"}>
                <ActivityLog game={game} />
              </div>
            </aside>
          </main>
          {choices && (
            <ChoiceDialog
              actions={choices}
              onAction={act}
              onClose={() => setChoices(null)}
            />
          )}
        </>
      )}
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(readRoute);
  useEffect(() => {
    const onPop = () => setSession(readRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const open = (s: Session) => {
    if (s.id)
      history.pushState(
        null,
        "",
        s.replay
          ? `/replays/${encodeURIComponent(s.id)}`
          : `/games/${encodeURIComponent(s.id)}`,
      );
    else history.pushState(null, "", "/");
    setSession(s);
  };
  const home = () => {
    history.pushState(null, "", "/");
    setSession(null);
  };
  return (
    <MotionConfig reducedMotion="user" transition={{ duration: 0.25 }}>
      {session ? (
        <GameScreen
          key={`${session.id || session.label}-${session.replay}`}
          session={session}
          open={open}
          home={home}
        />
      ) : (
        <HomeScreen open={open} />
      )}
    </MotionConfig>
  );
}
