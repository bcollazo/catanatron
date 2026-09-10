import { useRef } from "react";
import { SlidersHorizontal, X } from "lucide-react";
import { type GameState } from "./model";

export function GameSettingsButton({ game }: { game: GameState }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const settings = game.settings;
  const number = (value: unknown) =>
    typeof value === "number" && Number.isFinite(value)
      ? value
      : "Not recorded";
  return (
    <>
      <button
        aria-label="Inspect game settings"
        title="Game settings"
        onClick={() => dialog.current?.showModal()}
      >
        <SlidersHorizontal size={17} />
      </button>
      <dialog
        ref={dialog}
        className="choice-dialog"
        aria-labelledby="game-settings-title"
        onClick={(e) => {
          if (e.target === e.currentTarget) dialog.current?.close();
        }}
      >
        <div className="dialog-head">
          <h2 id="game-settings-title">Game settings</h2>
          <button
            aria-label="Close game settings"
            onClick={() => dialog.current?.close()}
          >
            <X size={20} />
          </button>
        </div>
        <dl className="game-settings">
          <div>
            <dt>Victory points to win</dt>
            <dd>{number(settings?.vps_to_win)}</dd>
          </div>
          <div>
            <dt>
              Discard threshold
              <small>
                On a 7, discard half your resource cards if your hand exceeds
                this count.
              </small>
            </dt>
            <dd>{number(settings?.discard_limit)}</dd>
          </div>
          <div>
            <dt>
              Friendly Robber
              <small>Protects players with fewer than 3 victory points.</small>
            </dt>
            <dd>
              {typeof settings?.friendly_robber === "boolean"
                ? settings.friendly_robber
                  ? "On"
                  : "Off"
                : "Not recorded"}
            </dd>
          </div>
          <div>
            <dt>Players</dt>
            <dd>{game.colors.length}</dd>
          </div>
          <div>
            <dt>Bot players</dt>
            <dd>{game.bot_colors.length}</dd>
          </div>
        </dl>
        <p className="muted">These settings are fixed for this game.</p>
      </dialog>
    </>
  );
}
