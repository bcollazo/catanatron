import { useEffect, useRef, type ReactNode } from "react";
import {
  Home,
  Building2,
  Route,
  Layers,
  ArrowLeftRight,
  Dices,
  ArrowRight,
  X,
  Shield,
  Sparkles,
} from "lucide-react";
import {
  actionKey,
  actionLabel,
  recordLabel,
  resourceColors,
  type Resource,
  title,
  type Action,
  type GameState,
} from "./model";

import { CardFace } from "./Resources";

function TradeCards({
  resources,
  label,
}: {
  resources: Resource[];
  label: string;
}) {
  return (
    <span className="trade-side">
      <span className="trade-caption">{label}</span>
      <span className="trade-cards">
        {resources.map((resource, i) => (
          <span
            key={i}
            className="hand-card"
            data-trade-card={resource}
            style={
              { "--resource": resourceColors[resource] } as React.CSSProperties
            }
          >
            <CardFace resource={resource} />
          </span>
        ))}
      </span>
    </span>
  );
}

export function ChoiceDialog({
  actions,
  onAction,
  onClose,
}: {
  actions: Action[];
  onAction: (a: Action) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const trading = actions[0]?.[1] === "MARITIME_TRADE";
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    ref.current?.showModal();
    return () => previous?.focus();
  }, []);
  return (
    <dialog
      ref={ref}
      className={`choice-dialog ${trading ? "trade-dialog" : ""}`}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dialog-head">
        <div>
          <span className="eyebrow">CHOOSE AN ACTION</span>
          <h2>
            {trading
              ? "Trade with the bank or a port"
              : title(actions[0]?.[1] || "Action").replace("Play ", "")}
          </h2>
        </div>
        <button aria-label="Close choices" onClick={onClose}>
          <X size={20} />
        </button>
      </div>
      <div className="choice-list">
        {actions.map((a) => (
          <button
            key={actionKey(a)}
            onClick={() => onAction(a)}
            aria-label={actionLabel(a)}
          >
            {a[1] === "MARITIME_TRADE" ? (
              <span className="trade-exchange" aria-hidden="true">
                <TradeCards
                  label="You give"
                  resources={a[2].slice(0, 4).filter(Boolean)}
                />
                <ArrowRight size={20} />
                <TradeCards label="You get" resources={[a[2][4]]} />
              </span>
            ) : (
              <>
                {actionLabel(a)}
                <ArrowRight size={16} />
              </>
            )}
          </button>
        ))}
      </div>
    </dialog>
  );
}

const buildItems = [
  {
    type: "BUILD_ROAD",
    label: "Build a Road",
    cost: "1 Wood · 1 Brick",
    icon: Route,
  },
  {
    type: "BUILD_SETTLEMENT",
    label: "Build a Settlement",
    cost: "1 Wood · 1 Brick · 1 Sheep · 1 Wheat",
    icon: Home,
  },
  {
    type: "BUILD_CITY",
    label: "Build a City",
    cost: "2 Wheat · 3 Ore",
    icon: Building2,
  },
  {
    type: "BUY_DEVELOPMENT_CARD",
    label: "Buy Development Card",
    cost: "1 Sheep · 1 Wheat · 1 Ore",
    icon: Layers,
  },
];
export function ActionDock({
  game,
  busy,
  mode,
  setMode,
  onAction,
  onChoices,
  botControls,
}: {
  game: GameState;
  busy: boolean;
  mode: string;
  setMode: (m: string) => void;
  onAction: (a: Action) => void;
  onChoices: (a: Action[]) => void;
  botControls?: ReactNode;
}) {
  const allowed = game.current_playable_actions;
  const human = !game.bot_colors.includes(game.current_color);
  const available = (type: string) => allowed.filter((a) => a[1] === type);
  const choose = (type: string) => {
    const options = available(type);
    if (type.startsWith("BUILD_") || type === "MOVE_ROBBER") {
      setMode(mode === type ? "" : type);
      return;
    }
    if (type === "MARITIME_TRADE") {
      onChoices(options);
      return;
    }
    if (options.length === 1) onAction(options[0]);
    else onChoices(options);
  };
  const primary = available("ROLL")[0] ?? available("END_TURN")[0];
  const prompt = game.winning_color
    ? `${title(game.winning_color)} wins the game`
    : !human
      ? `${title(game.current_color)} is planning a move`
      : game.is_initial_build_phase
        ? available("BUILD_SETTLEMENT").length
          ? "Place your settlement on a glowing intersection"
          : "Connect a road to your settlement"
        : game.current_prompt === "DISCARD"
          ? `Choose a resource to discard · ${game.current_discard_count} remaining`
          : game.current_prompt === "MOVE_ROBBER"
            ? "Choose a tile, then choose who to steal from"
            : allowed.length > 0 && allowed.every((a) => a[1] === "BUILD_ROAD")
              ? "Place your free road on a glowing edge"
              : mode
                ? `Choose a location to ${title(mode).toLowerCase()}`
                : primary?.[1] === "ROLL"
                  ? "Your turn. Roll the dice to collect resources."
                  : "Build, trade, or play a development card.";
  const last = game.action_records.at(-1);
  const recent = last
    ? `${title(last[0][0])} ${last[0][1] === "END_TURN" ? "ended their turn" : recordLabel(last)}. `
    : "";
  return (
    <section className="action-dock" aria-label="Game actions">
      <div className="turn-prompt">
        <span className={`status-dot ${busy ? "thinking" : ""}`} />
        <span
          role="status"
          aria-live="polite"
          title={busy ? "Applying move…" : recent + prompt}
        >
          {busy ? (
            "Applying move…"
          ) : (
            <>
              <span className="recent-move">{recent}</span>
              {prompt}
            </>
          )}
        </span>
        {mode && (
          <button className="text-button" onClick={() => setMode("")}>
            Cancel <X size={14} />
          </button>
        )}
      </div>
      {human && !game.winning_color && (
        <>
          <div className="dock-actions">
            {game.current_prompt === "DISCARD" ? (
              available("DISCARD_RESOURCE").map((a) => (
                <button
                  key={actionKey(a)}
                  disabled={busy}
                  onClick={() => onAction(a)}
                >
                  Discard {title(a[2])}
                </button>
              ))
            ) : (
              <>
                {buildItems
                  .filter(({ type }) => available(type).length > 0)
                  .map(({ type, label, cost, icon: Icon }) => (
                    <button
                      className={mode === type ? "selected" : ""}
                      key={type}
                      disabled={busy}
                      onClick={() => choose(type)}
                      title={cost}
                    >
                      <Icon size={20} />
                      <span>
                        {label}
                        <small>
                          {game.is_initial_build_phase
                            ? "Initial placement"
                            : cost}
                        </small>
                      </span>
                    </button>
                  ))}
                {available("MARITIME_TRADE").length > 0 && (
                  <button
                    disabled={busy}
                    onClick={() => choose("MARITIME_TRADE")}
                  >
                    <ArrowLeftRight size={20} />
                    <span>
                      Trade Resources<small>Choose a legal exchange</small>
                    </span>
                  </button>
                )}
                {[
                  "PLAY_KNIGHT_CARD",
                  "PLAY_MONOPOLY",
                  "PLAY_YEAR_OF_PLENTY",
                  "PLAY_ROAD_BUILDING",
                ]
                  .filter((type) => available(type).length > 0)
                  .map((type) => (
                    <button
                      key={type}
                      disabled={busy}
                      onClick={() => choose(type)}
                    >
                      {type.includes("KNIGHT") ? (
                        <Shield size={20} />
                      ) : (
                        <Sparkles size={20} />
                      )}
                      <span>{title(type).replace(" Card", "")}</span>
                    </button>
                  ))}
              </>
            )}
            {available("MOVE_ROBBER").length > 0 && (
              <button
                className="primary"
                disabled={busy}
                onClick={() => setMode("MOVE_ROBBER")}
              >
                <Shield size={18} /> Move robber
              </button>
            )}
            {primary && (
              <button
                className="primary advance"
                disabled={busy}
                onClick={() => onAction(primary)}
              >
                {primary[1] === "ROLL" ? (
                  <Dices size={21} />
                ) : (
                  <ArrowRight size={21} />
                )}{" "}
                {primary[1] === "ROLL" ? "Roll dice" : "End turn"}
              </button>
            )}
          </div>
        </>
      )}
      {(!human || game.winning_color) && (
        <div className="dock-idle">{botControls}</div>
      )}
    </section>
  );
}
