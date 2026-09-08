import type { GameState, PlayerState } from "../utils/api.types";

function makePlayerState(playerIndex: number): PlayerState {
  const key = `P${playerIndex}`;
  return {
    [`${key}_ACTUAL_VICTORY_POINTS`]: 0,
    [`${key}_BRICK_IN_HAND`]: 0,
    [`${key}_HAS_ARMY`]: false,
    [`${key}_HAS_ROAD`]: false,
    [`${key}_HAS_ROLLED`]: false,
    [`${key}_KNIGHT_IN_HAND`]: 0,
    [`${key}_LONGEST_ROAD_LENGTH`]: 0,
    [`${key}_MONOPOLY_IN_HAND`]: 0,
    [`${key}_ORE_IN_HAND`]: 0,
    [`${key}_PLAYED_KNIGHT`]: 0,
    [`${key}_ROAD_BUILDING_IN_HAND`]: 0,
    [`${key}_SHEEP_IN_HAND`]: 0,
    [`${key}_VICTORY_POINT_IN_HAND`]: 0,
    [`${key}_WHEAT_IN_HAND`]: 0,
    [`${key}_WOOD_IN_HAND`]: 0,
    [`${key}_YEAR_OF_PLENTY_IN_HAND`]: 0,
  };
}

/**
 * Representative of the JSON emitted by catanatron.serialization.web_view:
 * the state document, plus what a browser needs on top of it.
 *
 * In particular, nodes are serialized as an object, a generic port has a
 * null resource, water tiles are present, and winning_color is null until
 * the game ends.
 */
export function makeGameState(
  overrides: Partial<GameState> = {}
): GameState {
  return {
    tiles: [
      {
        coordinate: [0, 0, 0],
        tile: {
          id: 0,
          type: "RESOURCE_TILE",
          resource: "WHEAT",
          number: 11,
        },
      },
      {
        coordinate: [1, -1, 0],
        tile: {
          id: 19,
          type: "PORT",
          direction: "WEST",
          resource: null,
        },
      },
      {
        coordinate: [2, -2, 0],
        tile: {
          type: "WATER",
        },
      },
    ],
    adjacent_tiles: {
      "0": [
        {
          id: 0,
          type: "RESOURCE_TILE",
          resource: "WHEAT",
          number: 11,
        },
      ],
    },
    bot_colors: ["RED"],
    colors: ["RED", "BLUE"],
    current_color: "BLUE",
    winning_color: null,
    current_prompt: "PLAY_TURN",
    player_state: {
      ...makePlayerState(0),
      ...makePlayerState(1),
      P1_WOOD_IN_HAND: 2,
    },
    action_records: [],
    robber_coordinate: [0, 0, 0],
    current_discard_count: 0,
    nodes: {
      "0": {
        id: 0,
        tile_coordinate: [0, 0, 0],
        direction: "SOUTHWEST",
        building: "SETTLEMENT",
        color: "RED",
      },
    },
    edges: [
      {
        id: [0, 1],
        color: "RED",
        direction: "WEST",
        tile_coordinate: [0, 0, 0],
      },
    ],
    current_playable_actions: [["BLUE", "ROLL", null]],
    is_initial_build_phase: false,
    longest_roads_by_player: {
      RED: 1,
      BLUE: 0,
    },
    state_index: 12,

    // The state document half of the payload.
    schema_version: 2,
    game: {
      id: "game-123",
      vps_to_win: 10,
      discard_limit: 7,
      friendly_robber: false,
    },
    num_turns: 6,
    current_player_index: 1,
    current_turn_index: 1,
    resource_freqdeck: [19, 19, 19, 19, 19],
    development_listdeck: { KNIGHT: 14, VICTORY_POINT: 5 },
    ...overrides,
  };
}
