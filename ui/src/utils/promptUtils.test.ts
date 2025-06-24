import { describe, test, expect } from "vitest";
import {
  getTileString,
  getShortTileString,
  findTileById,
  findTileByCoordinate,
  humanizeAction,
  humanizeTradeAction,
} from "./promptUtils";
import type { GameState, GameAction } from "./api.types";

describe("getTileString", () => {
  test("Resource", () => {
    expect(
      getTileString({
        coordinate: [0, 0, 0],
        tile: {
          id: 0,
          type: "RESOURCE_TILE",
          resource: "BRICK",
          number: 6,
        },
      })
    ).toEqual("6 BRICK");
  });
  test("Desert", () => {
    expect(
      getTileString({
        coordinate: [1, 1, 1],
        tile: {
          id: 1,
          type: "DESERT",
        },
      })
    ).toEqual("THE DESERT");
  });
  test("Other", () => {
    expect(() =>
      getTileString({
        coordinate: [0, 0, 1],
        tile: {
          id: 2,
          type: "PORT",
          resource: "WHEAT",
          direction: "NORTHEAST",
        },
      })
    ).toThrowError("getTileString() only works on Desert or Resource tiles");
  });
});

describe("getShortTileString", () => {
  test("Resource", () => {
    expect(
      getShortTileString({
        id: 0,
        type: "RESOURCE_TILE",
        resource: "BRICK",
        number: 6,
      })
    ).toEqual("6");
  });
  test("Desert", () => {
    expect(
      getShortTileString({
        id: 1,
        type: "DESERT",
      })
    ).toEqual("DESERT");
  });
  test("Port", () => {
    expect(
      getShortTileString({
        id: 2,
        type: "PORT",
        resource: "WHEAT",
        direction: "NORTHEAST",
      })
    ).toEqual("PORT");
  });
});

describe("findTileById", () => {
  const gameState = {
    tiles: [
      {
        coordinate: [0, 0, 0],
        tile: {
          id: 0,
          type: "RESOURCE_TILE",
          resource: "BRICK",
          number: 6,
        },
      },
    ],
  };
  const expected = gameState.tiles[0];
  test("found", () => {
    expect(findTileById(gameState as GameState, 0)).toEqual(expected);
  });
  test("not found", () => {
    expect(findTileById(gameState as GameState, 1)).toBeUndefined();
  });
});

describe("findTileByCoordinate", () => {
  const gameState = {
    tiles: [
      {
        coordinate: [0, 0, 0],
        tile: {
          id: 0,
          type: "RESOURCE_TILE",
          resource: "BRICK",
          number: 6,
        },
      },
    ],
  };
  const expected = gameState.tiles[0];
  test("found", () => {
    expect(findTileByCoordinate(gameState as GameState, [0, 0, 0])).toEqual(
      expected
    );
  });
  test("not found", () => {
    expect(() =>
      findTileByCoordinate(gameState as GameState, [0, 0, 1])
    ).toThrowError("Tile not found for coordinate: [0,0,1]");
  });
});

describe("humanizeAction", () => {
  const baseGameState = {
    bot_colors: ["RED", "BLUE"],
    adjacent_tiles: {
      0: [
        { id: 0, type: "RESOURCE_TILE", resource: "BRICK", number: 6 },
        { id: 1, type: "DESERT" },
      ],
      1: [
        { id: 1, type: "DESERT" },
        { id: 2, type: "RESOURCE_TILE", resource: "WHEAT", number: 8 },
      ],
      2: [
        { id: 2, type: "RESOURCE_TILE", resource: "WHEAT", number: 8 },
        { id: 0, type: "RESOURCE_TILE", resource: "BRICK", number: 6 },
      ],
      A: [
        { id: 0, type: "RESOURCE_TILE", resource: "BRICK", number: 6 },
        { id: 1, type: "DESERT" },
      ],
      B: [
        { id: 1, type: "DESERT" },
        { id: 2, type: "RESOURCE_TILE", resource: "WHEAT", number: 8 },
      ],
    },
    tiles: [
      {
        coordinate: [0, 0, 0],
        tile: { id: 0, type: "RESOURCE_TILE", resource: "BRICK", number: 6 },
      },
      { coordinate: [1, 1, 1], tile: { id: 1, type: "DESERT" } },
      {
        coordinate: [2, 2, 2],
        tile: { id: 2, type: "RESOURCE_TILE", resource: "WHEAT", number: 8 },
      },
    ],
  } as unknown as GameState;

  test("ROLL action", () => {
    expect(humanizeAction(baseGameState, ["RED", "ROLL", [3, 4]])).toBe(
      "BOT ROLLED A 7"
    );
  });

  test("DISCARD action", () => {
    expect(humanizeAction(baseGameState, ["ORANGE", "DISCARD", null])).toBe(
      "YOU DISCARDED"
    );
  });

  test("BUY_DEVELOPMENT_CARD action", () => {
    expect(
      humanizeAction(baseGameState, ["BLUE", "BUY_DEVELOPMENT_CARD", null])
    ).toBe("BOT BOUGHT DEVELOPMENT CARD");
  });

  test("BUILD_SETTLEMENT action", () => {
    expect(humanizeAction(baseGameState, ["RED", "BUILD_SETTLEMENT", 0])).toBe(
      "BOT BUILT SETTLEMENT ON 6-DESERT"
    );
  });

  test("BUILD_CITY action", () => {
    expect(humanizeAction(baseGameState, ["ORANGE", "BUILD_CITY", 1])).toBe(
      "YOU BUILT CITY ON DESERT-8"
    );
  });

  test("BUILD_ROAD action", () => {
    expect(humanizeAction(baseGameState, ["RED", "BUILD_ROAD", [0, 0]])).toBe(
      "BOT BUILT ROAD ON 6-DESERT"
    );
  });

  test("PLAY_KNIGHT_CARD action", () => {
    expect(
      humanizeAction(baseGameState, ["BLUE", "PLAY_KNIGHT_CARD", null])
    ).toBe("BOT PLAYED KNIGHT CARD");
  });

  test("PLAY_ROAD_BUILDING action", () => {
    expect(
      humanizeAction(baseGameState, ["ORANGE", "PLAY_ROAD_BUILDING", null])
    ).toBe("YOU PLAYED ROAD BUILDING");
  });

  test("PLAY_MONOPOLY action", () => {
    expect(
      humanizeAction(baseGameState, ["RED", "PLAY_MONOPOLY", "BRICK"])
    ).toBe("BOT MONOPOLIZED BRICK");
  });

  test("PLAY_YEAR_OF_PLENTY action with two resources", () => {
    expect(
      humanizeAction(baseGameState, [
        "ORANGE",
        "PLAY_YEAR_OF_PLENTY",
        ["BRICK", "WHEAT"],
      ])
    ).toBe("YOU PLAYED YEAR OF PLENTY. CLAIMED BRICK AND WHEAT");
  });

  test("PLAY_YEAR_OF_PLENTY action with one resource", () => {
    expect(
      humanizeAction(baseGameState, [
        "ORANGE",
        "PLAY_YEAR_OF_PLENTY",
        ["BRICK"],
      ])
    ).toBe("YOU PLAYED YEAR OF PLENTY. CLAIMED BRICK");
  });

  test("MOVE_ROBBER action with stolen resource", () => {
    const action: GameAction = [
      "RED",
      "MOVE_ROBBER",
      [[0, 0, 0], "BLUE", "BRICK"],
    ];
    expect(humanizeAction(baseGameState, action)).toBe(
      "BOT ROBBED 6 BRICK (STOLE BRICK)"
    );
  });

  test("MOVE_ROBBER action without stolen resource", () => {
    const action: GameAction = [
      "RED",
      "MOVE_ROBBER",
      [[0, 0, 0], undefined, undefined],
    ];
    expect(humanizeAction(baseGameState, action)).toBe("BOT ROBBED 6 BRICK");
  });

  test("MARITIME_TRADE action", () => {
    const action: GameAction = [
      "ORANGE",
      "MARITIME_TRADE",
      ["BRICK", "BRICK", "BRICK", null, "WHEAT"],
    ];
    expect(humanizeAction(baseGameState, action)).toBe(
      "YOU TRADED 3 BRICK => WHEAT"
    );
  });

  test("END_TURN action", () => {
    expect(humanizeAction(baseGameState, ["RED", "END_TURN", null])).toBe(
      "BOT ENDED TURN"
    );
  });

  test("Unknown action type throws", () => {
    expect(() =>
      humanizeAction(baseGameState, [
        "RED",
        // @ts-expect-error
        "UNKNOWN_ACTION",
      ])
    ).toThrowError("Unknown action type: UNKNOWN_ACTION");
  });

  test("ROLL action with missing outcomes throws", () => {
    // @ts-expect-error
    expect(() => humanizeAction(baseGameState, ["RED", "ROLL"])).toThrowError(
      "did not get Rolling outcomes back from Server! output"
    );
  });
});

describe("humanizeTradeAction", () => {
  test("returns correct string for 3:1 trade", () => {
    const action: GameAction = [
      "RED",
      "MARITIME_TRADE",
      ["BRICK", "BRICK", "BRICK", null, "WHEAT"],
    ];
    expect(humanizeTradeAction(action)).toBe("3 BRICK => WHEAT");
  });

  test("returns correct string for 2:1 trade", () => {
    const action: GameAction = [
      "RED",
      "MARITIME_TRADE",
      ["WHEAT", "WHEAT", null, null, "BRICK"],
    ];
    expect(humanizeTradeAction(action)).toBe("2 WHEAT => BRICK");
  });

  test("returns correct string for 4:1 trade", () => {
    const action: GameAction = [
      "RED",
      "MARITIME_TRADE",
      ["ORE", "ORE", "ORE", "ORE", "WOOD"],
    ];
    expect(humanizeTradeAction(action)).toBe("4 ORE => WOOD");
  });

  test("returns correct string for 1:1 trade", () => {
    const action: GameAction = [
      "RED",
      "MARITIME_TRADE",
      ["BRICK", null, null, null, "ORE"],
    ];
    expect(humanizeTradeAction(action)).toBe("1 BRICK => ORE");
  });
});
