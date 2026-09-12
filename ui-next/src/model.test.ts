import { describe, expect, it } from "vitest";
import snapshots from "../public/example-game.json";
import rustReplay from "../public/example-rust-ab2-vs-ab2.json";
import { parseReplay, actionLabel, nodePoint, type Action } from "./model";

describe("CLI compatibility", () => {
  it("opens real GameEncoder output as a final snapshot or a timeline", () => {
    const frames = parseReplay(JSON.stringify(snapshots));
    expect(frames).toHaveLength(81);
    expect(parseReplay(JSON.stringify(snapshots.at(-1)))[0].state_index).toBe(
      80,
    );
    expect(
      parseReplay(JSON.stringify({ states: [snapshots[3], snapshots[0]] })).map(
        (s) => s.state_index,
      ),
    ).toEqual([0, 3]);
  });
  it("rejects malformed files with an actionable message", () => {
    expect(() => parseReplay("not json")).toThrow("valid JSON");
    expect(() => parseReplay("{}")).toThrow("snapshot");
    expect(() => parseReplay("[]")).toThrow("snapshot");
    expect(() =>
      parseReplay(JSON.stringify({ ...snapshots[0], bot_colors: undefined })),
    ).toThrow("snapshot");
    expect(() =>
      parseReplay(
        JSON.stringify({
          ...snapshots[0],
          tiles: [
            {
              coordinate: [0, 0, 0],
              tile: { type: "RESOURCE_TILE", resource: "BANANA" },
            },
          ],
        }),
      ),
    ).toThrow("snapshot");
  });
  it("places every edge of a real map between two neighboring intersections", () => {
    const game = parseReplay(JSON.stringify(snapshots[0]))[0];
    for (const edge of game.edges) {
      const p = nodePoint(game.nodes[edge.id[0]]),
        q = nodePoint(game.nodes[edge.id[1]]);
      expect(Math.hypot(p[0] - q[0], p[1] - q[1])).toBeCloseTo(58, 5);
    }
  });
  it("labels bank and port trades without counting padding as cards", () => {
    expect(
      actionLabel([
        "RED",
        "MARITIME_TRADE",
        ["WOOD", "WOOD", null, null, "ORE"],
      ]),
    ).toBe("2 Wood → 1 Ore");
    expect(actionLabel(["RED", "PLAY_YEAR_OF_PLENTY", ["SHEEP"]])).toBe(
      "Sheep",
    );
    expect(actionLabel(["RED", "MOVE_ROBBER", [[0, 0, 0], "BLUE"]])).toBe(
      "Steal from Blue",
    );
  });
});

describe("Rust engine replay import", () => {
  // Both seats were decided entirely by the Rust engine's alpha-beta search
  // (rust/crates/bench/src/bin/export-tournament-replay.rs), then
  // force-replayed through the real Python engine
  // (rust/tools/replay_trace_for_ui.py) so every snapshot is
  // catanatron.serialization.web_view() -- schema_version 2, not the
  // retired GameEncoder. This is the same document shape live/DB-backed
  // play already uses; nothing in model.ts changed to read it.
  it("opens a Rust-vs-Rust alpha-beta game exported via web_view()", () => {
    const frames = parseReplay(JSON.stringify(rustReplay));
    expect(frames.length).toBeGreaterThan(100);
    expect(frames[0].state_index).toBe(0);
    expect(frames.at(-1)!.state_index).toBe(frames.length - 1);
    expect(frames.at(-1)!.winning_color).not.toBeNull();
    expect(frames.at(-1)!.colors).toEqual(["BLUE", "RED"]);
  });
});
