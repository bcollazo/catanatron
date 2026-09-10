import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { Board } from "./Board";
import { parseReplay, portGeometry, type Action } from "./model";
import snapshots from "../public/example-game.json";
const game = parseReplay(JSON.stringify(snapshots[0]))[0];
beforeAll(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  window.matchMedia = vi.fn().mockReturnValue({
    matches: false,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
});
afterEach(cleanup);
describe("board interaction", () => {
  it("connects every port to the exact engine-defined coastal nodes", () => {
    // From the Python engine's Port.nodes and PORT_DIRECTION_TO_NODEREFS.
    const expected: Record<string, number[]> = {
      "3,-3,0": [25, 26],
      "1,-3,2": [28, 29],
      "-1,-2,3": [32, 33],
      "-3,0,3": [35, 36],
      "-3,2,1": [38, 39],
      "-2,3,-1": [40, 44],
      "0,3,-3": [47, 45],
      "2,1,-3": [48, 49],
      "3,-1,-2": [52, 53],
    };
    const ports = game.tiles.filter((t) => t.tile.type === "PORT");
    expect(ports).toHaveLength(9);
    for (const port of ports)
      expect(portGeometry(game, port)?.nodes).toEqual(
        expected[port.coordinate.join(",")],
      );
    const { container } = render(<Board game={game} animated={false} />);
    expect(container.querySelectorAll(".port-link")).toHaveLength(18);
    expect(container.querySelector(".port-connection polygon")).toBeNull();
    expect(container.querySelector(".lucide-anchor")).toBeNull();
    expect(
      container.querySelector(".port-connection .lucide-stone"),
    ).not.toBeNull();
    const tileText = Array.from(container.querySelectorAll(".land-tile text"))
      .map((n) => n.textContent)
      .join(" ");
    expect(tileText).not.toMatch(/Wood|Brick|Sheep|Wheat|Ore/);
    // The resource names remain accessible when tile labels are visually removed.
    expect(
      screen.getAllByRole("button", { name: /Ore · rolls/ }).length,
    ).toBeGreaterThan(0);
  });
  it("keeps pips within number tokens and leaves the desert without a resource icon", () => {
    const { container } = render(<Board game={game} animated={false} />);
    const desert = container.querySelector('[data-resource="DESERT"]')!;
    expect(desert.querySelector("svg")).toBeNull();
    expect(desert.querySelector("polygon")?.getAttribute("fill")).toBe(
      "#c29a63",
    );
    const land = container.querySelector('.land-tile[data-resource="WOOD"]')!;
    const token = land.querySelector("circle")!;
    const pips = land.querySelector(".tile-pips")!;
    expect(Number(pips.getAttribute("y"))).toBeLessThan(
      Number(token.getAttribute("cy")) + Number(token.getAttribute("r")),
    );
    expect(
      container
        .querySelector('.port-badge circle[fill="var(--panel)"]')
        ?.getAttribute("r"),
    ).toBe("16");
    for (const port of game.tiles.filter((t) => t.tile.type === "PORT")) {
      const { badge, ends } = portGeometry(game, port)!;
      const midpoint = [
        (ends[0][0] + ends[1][0]) / 2,
        (ends[0][1] + ends[1][1]) / 2,
      ];
      expect(
        Math.hypot(badge[0] - midpoint[0], badge[1] - midpoint[1]),
      ).toBeCloseTo(26);
    }
  });
  it("allows keyboard placement of a legal initial settlement", () => {
    const action = game.current_playable_actions[0];
    const onAction = vi.fn();
    render(
      <Board
        game={game}
        actions={[action]}
        onAction={onAction}
        animated={false}
      />,
    );
    const target = screen.getByRole("button", {
      name: `Build Settlement at node ${action[2]}`,
    });
    fireEvent.keyDown(target, { key: "Enter" });
    expect(onAction).toHaveBeenCalledWith(action);
  });
  it("leaves board building targets inactive in replay mode", () => {
    render(<Board game={game} animated={false} />);
    expect(
      screen.queryByRole("button", { name: /Build Settlement at node/ }),
    ).toBeNull();
  });
  it("asks for a victim when the selected tile has multiple legal robber actions", () => {
    const coord = game.tiles.find(
      (t) => t.tile.type === "RESOURCE_TILE",
    )!.coordinate;
    const options: Action[] = [
      ["RED", "MOVE_ROBBER", [coord, "BLUE"]],
      ["RED", "MOVE_ROBBER", [coord, "ORANGE"]],
    ];
    const onChoices = vi.fn(),
      onAction = vi.fn();
    render(
      <Board
        game={game}
        actions={options}
        onChoices={onChoices}
        onAction={onAction}
        animated={false}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Move robber to/ }));
    expect(onChoices).toHaveBeenCalledWith(options);
    expect(onAction).not.toHaveBeenCalled();
  });
});
