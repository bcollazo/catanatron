import { afterEach, expect, it } from "vitest";
import { cleanup, render } from "@testing-library/react";
import { GameSettingsButton } from "./GameSettings";
import { parseReplay } from "./model";
import snapshot from "../public/example-preview.json";
afterEach(cleanup);
const game = parseReplay(JSON.stringify(snapshot))[0];
it("displays the recorded rules, including a disabled Friendly Robber", () => {
  const { container } = render(
    <GameSettingsButton
      game={{
        ...game,
        settings: { vps_to_win: 14, discard_limit: 9, friendly_robber: false },
      }}
    />,
  );
  expect(
    [...container.querySelectorAll("dd")].map((el) => el.textContent),
  ).toEqual([
    "14",
    "9",
    "Off",
    String(game.colors.length),
    String(game.bot_colors.length),
  ]);
  expect(container.querySelectorAll("input, select")).toHaveLength(0);
});
it("does not invent rules for older snapshots", () => {
  const { container } = render(<GameSettingsButton game={game} />);
  expect(
    [...container.querySelectorAll("dd")].filter(
      (el) => el.textContent === "Not recorded",
    ),
  ).toHaveLength(3);
});
