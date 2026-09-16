import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { ResourceHand, productionSource } from "./Resources";
import { parseReplay, type GameState, nodePoint, center, same } from "./model";
import snapshots from "../public/example-game.json";
const base = parseReplay(JSON.stringify(snapshots[0]))[0];
const withHand = (wood: number, ore: number): GameState => ({
  ...base,
  player_state: {
    ...base.player_state,
    P0_WOOD_IN_HAND: wood,
    P0_BRICK_IN_HAND: 0,
    P0_SHEEP_IN_HAND: 0,
    P0_WHEAT_IN_HAND: 0,
    P0_ORE_IN_HAND: ore,
  },
});
afterEach(() => {
  cleanup();
  localStorage.clear();
});
describe("physical resource cards", () => {
  it("renders actual copies and no placeholders for absent resources", () => {
    render(<ResourceHand game={withHand(3, 2)} color={base.colors[0]} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(5);
    expect(screen.getAllByRole("listitem", { name: "Wood" })).toHaveLength(3);
    expect(screen.getAllByRole("listitem", { name: "Ore" })).toHaveLength(2);
    expect(screen.queryByRole("listitem", { name: "Sheep" })).toBeNull();
  });
  it("adds and removes individual cards as inventory changes", async () => {
    const { rerender } = render(
      <ResourceHand game={withHand(2, 1)} color={base.colors[0]} />,
    );
    rerender(
      <ResourceHand
        game={{ ...withHand(4, 1), state_index: 1 }}
        color={base.colors[0]}
      />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(5);
    rerender(
      <ResourceHand
        game={{ ...withHand(1, 1), state_index: 2 }}
        color={base.colors[0]}
      />,
    );
    await waitFor(() =>
      expect(screen.getAllByRole("listitem")).toHaveLength(2),
    );
  });
  it("includes individual development cards in the main hand and removes played copies", async () => {
    const game = withHand(1, 0);
    game.player_state = {
      ...game.player_state,
      P0_KNIGHT_IN_HAND: 2,
      P0_MONOPOLY_IN_HAND: 1,
      P0_YEAR_OF_PLENTY_IN_HAND: 1,
      P0_ROAD_BUILDING_IN_HAND: 1,
      P0_VICTORY_POINT_IN_HAND: 1,
    };
    const { rerender } = render(
      <ResourceHand game={game} color={base.colors[0]} />,
    );
    expect(screen.getAllByRole("listitem")).toHaveLength(7);
    expect(screen.getAllByRole("listitem", { name: "Knight" })).toHaveLength(2);
    for (const name of [
      "Monopoly",
      "Year Of Plenty",
      "Road Building",
      "Victory Point",
    ])
      expect(screen.getByRole("listitem", { name })).toBeTruthy();
    rerender(
      <ResourceHand
        game={{
          ...game,
          state_index: game.state_index + 1,
          player_state: { ...game.player_state, P0_KNIGHT_IN_HAND: 1 },
        }}
        color={base.colors[0]}
      />,
    );
    await waitFor(() =>
      expect(screen.getAllByRole("listitem", { name: "Knight" })).toHaveLength(
        1,
      ),
    );
  });
  it("does not duplicate development cards in the compact player-panel hand", () => {
    const game = withHand(1, 0);
    game.player_state = { ...game.player_state, P0_KNIGHT_IN_HAND: 2 };
    render(<ResourceHand game={game} color={base.colors[0]} compact />);
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    expect(screen.queryByRole("listitem", { name: "Knight" })).toBeNull();
  });
  it("identifies an owned, producing tile as the card animation origin", () => {
    const frames = parseReplay(JSON.stringify(snapshots));
    const frame = frames.find(
      (s) =>
        s.action_records.at(-1)?.[0][1] === "ROLL" &&
        s.action_records.at(-1)![1][0] + s.action_records.at(-1)![1][1] !== 7,
    )!;
    let origins = 0;
    for (const color of frame.colors)
      for (const resource of [
        "WOOD",
        "BRICK",
        "SHEEP",
        "WHEAT",
        "ORE",
      ] as const) {
        const coordinate = productionSource(frame, color, resource);
        if (!coordinate) continue;
        origins++;
        expect(same(coordinate, frame.robber_coordinate)).toBe(false);
        expect(
          Object.values(frame.nodes).some(
            (n) =>
              n.color === color &&
              n.building &&
              Math.abs(
                Math.hypot(
                  nodePoint(n)[0] - center(coordinate)[0],
                  nodePoint(n)[1] - center(coordinate)[1],
                ) - 58,
              ) < 0.001,
          ),
        ).toBe(true);
      }
    expect(origins).toBeGreaterThan(0);
  });
});
