import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { ActionDock, ChoiceDialog } from "./ActionDock";
import { parseReplay, type Action } from "./model";
import snapshots from "../public/example-game.json";
const base = {
  ...parseReplay(JSON.stringify(snapshots[0]))[0],
  current_color: "RED" as const,
  bot_colors: [],
  is_initial_build_phase: false,
};
afterEach(cleanup);
describe("human action controls", () => {
  it.each([
    ["Cards", "Year Of Plenty", "PLAY_YEAR_OF_PLENTY", ["SHEEP"]],
    ["Cards", "Monopoly", "PLAY_MONOPOLY", "WOOD"],
    ["Cards", "Knight", "PLAY_KNIGHT_CARD", null],
    ["Cards", "Road Building", "PLAY_ROAD_BUILDING", null],
  ])(
    "submits the exact server payload for %s / %s",
    (_category, label, type, payload) => {
      const a: Action = ["RED", type, payload];
      const onAction = vi.fn();
      render(
        <ActionDock
          game={{ ...base, current_playable_actions: [a] }}
          busy={false}
          mode=""
          setMode={vi.fn()}
          onAction={onAction}
          onChoices={vi.fn()}
        />,
      );
      fireEvent.click(screen.getByRole("button", { name: new RegExp(label) }));
      expect(onAction).toHaveBeenCalledWith(a);
    },
  );
  it("shows only legal actions together and updates them after a move", () => {
    const setMode = vi.fn();
    const props = {
      busy: false,
      mode: "",
      setMode,
      onAction: vi.fn(),
      onChoices: vi.fn(),
    };
    const actions: Action[] = [
      ["RED", "BUILD_ROAD", [0, 1]],
      ["RED", "MARITIME_TRADE", ["WOOD", "WOOD", null, null, "ORE"]],
      ["RED", "PLAY_KNIGHT_CARD", null],
      ["RED", "END_TURN", null],
    ];
    const { rerender } = render(
      <ActionDock
        {...props}
        game={{ ...base, current_playable_actions: actions }}
      />,
    );
    expect(screen.getAllByRole("button")).toHaveLength(4);
    expect(
      screen.getByRole("button", { name: /Trade Resources/ }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Play Knight" })).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: /Settlement|City|Develop/ }),
    ).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^Build a Road / }));
    expect(setMode).toHaveBeenCalledWith("BUILD_ROAD");
    rerender(
      <ActionDock
        {...props}
        game={{
          ...base,
          current_playable_actions: [["RED", "END_TURN", null]],
        }}
      />,
    );
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "End turn" })).toBeTruthy();
  });
  it("opens a choice when multiple development-card outcomes are legal", () => {
    const options: Action[] = [
      ["RED", "PLAY_YEAR_OF_PLENTY", ["WOOD"]],
      ["RED", "PLAY_YEAR_OF_PLENTY", ["ORE"]],
    ];
    const onChoices = vi.fn();
    render(
      <ActionDock
        game={{ ...base, current_playable_actions: options }}
        busy={false}
        mode=""
        setMode={vi.fn()}
        onAction={vi.fn()}
        onChoices={onChoices}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Play Year Of Plenty" }),
    );
    expect(onChoices).toHaveBeenCalledWith(options);
  });
  it.each([2, 3, 4])(
    "shows each card in a %s:1 trade and submits the displayed exchange",
    (count) => {
      HTMLDialogElement.prototype.showModal = vi.fn();
      const a: Action = [
        "RED",
        "MARITIME_TRADE",
        [
          ...Array(count).fill("BRICK"),
          ...Array(4 - count).fill(null),
          "WHEAT",
        ],
      ];
      const onAction = vi.fn();
      const { container } = render(
        <ChoiceDialog actions={[a]} onAction={onAction} onClose={vi.fn()} />,
      );
      expect(
        container.querySelectorAll('[data-trade-card="BRICK"]'),
      ).toHaveLength(count);
      expect(
        container.querySelectorAll('[data-trade-card="WHEAT"]'),
      ).toHaveLength(1);
      fireEvent.click(container.querySelector(".choice-list button")!);
      expect(onAction).toHaveBeenCalledWith(a);
    },
  );
  it("previews a trade even when only one exchange is legal", () => {
    const a: Action = [
      "RED",
      "MARITIME_TRADE",
      ["WOOD", "WOOD", null, null, "ORE"],
    ];
    const onAction = vi.fn(),
      onChoices = vi.fn();
    render(
      <ActionDock
        game={{ ...base, current_playable_actions: [a] }}
        busy={false}
        mode=""
        setMode={vi.fn()}
        onAction={onAction}
        onChoices={onChoices}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Trade Resources/ }));
    expect(onAction).not.toHaveBeenCalled();
    expect(onChoices).toHaveBeenCalledWith([a]);
  });
  it("combines the last move and next instruction into one status", () => {
    render(
      <ActionDock
        game={{
          ...base,
          action_records: [[["BLUE", "END_TURN", null], null]],
          current_playable_actions: [["RED", "ROLL", null]],
        }}
        busy={false}
        mode=""
        setMode={vi.fn()}
        onAction={vi.fn()}
        onChoices={vi.fn()}
      />,
    );
    expect(screen.getAllByRole("status")).toHaveLength(1);
    expect(screen.getByRole("status").textContent).toBe(
      "Blue ended their turn. Your turn. Roll the dice to collect resources.",
    );
  });
  it("preserves the selected victim instead of silently selecting the first one", () => {
    HTMLDialogElement.prototype.showModal = vi.fn();
    const options: Action[] = [
      ["RED", "MOVE_ROBBER", [[0, 0, 0], "BLUE"]],
      ["RED", "MOVE_ROBBER", [[0, 0, 0], "ORANGE"]],
    ];
    const onAction = vi.fn();
    render(
      <ChoiceDialog actions={options} onAction={onAction} onClose={vi.fn()} />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Steal from Orange", hidden: true }),
    );
    expect(onAction).toHaveBeenCalledWith(options[1]);
  });
});
