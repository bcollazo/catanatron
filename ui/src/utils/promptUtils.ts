import type {
  Tile,
  PlacedTile,
  GameActionRecord,
  MaritimeTradeAction,
  BuildCityAction,
  BuildRoadAction,
  PlayYearOfPlentyAction,
  MoveRobberAction,
} from "./api.types";
import type { GameState } from "./api.types";

export function humanizeActionRecord(
  gameState: GameState,
  actionRecord: GameActionRecord
) {
  const botColors = gameState.bot_colors;
  const action = actionRecord[0];
  const player = botColors.includes(action[0]) ? "BOT" : "YOU";
  switch (actionRecord[0][1]) {
    case "ROLL": {
      const action = actionRecord[1] as [number, number];
      return `${player} ROLLED A ${action[0] + action[1]}`;
    }
    case "DISCARD":
      return `${player} DISCARDED`;
    case "BUY_DEVELOPMENT_CARD":
      return `${player} BOUGHT DEVELOPMENT CARD`;
    case "BUILD_SETTLEMENT":
    case "BUILD_CITY": {
      const action = actionRecord[0] as BuildCityAction;
      const parts = action[1].split("_");
      const building = parts[parts.length - 1];
      const tileId = action[2];
      const tiles = gameState.adjacent_tiles[tileId];
      const tileString = tiles.map(getShortTileString).join("-");
      return `${player} BUILT ${building} ON ${tileString}`;
    }
    case "BUILD_ROAD": {
      const action = actionRecord[0] as BuildRoadAction;
      const edge = action[2];
      const a = gameState.adjacent_tiles[edge[0]].map((t) => t.id);
      const b = gameState.adjacent_tiles[edge[1]].map((t) => t.id);
      const intersection = a.filter((t) => b.includes(t));
      const tiles = intersection.map(
        (tileId) => findTileById(gameState, tileId).tile
      );
      const edgeString = tiles.map(getShortTileString).join("-");
      return `${player} BUILT ROAD ON ${edgeString}`;
    }
    case "PLAY_KNIGHT_CARD": {
      return `${player} PLAYED KNIGHT CARD`;
    }
    case "PLAY_ROAD_BUILDING": {
      return `${player} PLAYED ROAD BUILDING`;
    }
    case "PLAY_MONOPOLY": {
      return `${player} MONOPOLIZED ${action[2]}`;
    }
    case "PLAY_YEAR_OF_PLENTY": {
      const action = actionRecord[0] as PlayYearOfPlentyAction;
      const firstResource = action[2][0];
      const secondResource = action[2][1];
      if (secondResource) {
        return `${player} PLAYED YEAR OF PLENTY. CLAIMED ${firstResource} AND ${secondResource}`;
      } else {
        return `${player} PLAYED YEAR OF PLENTY. CLAIMED ${firstResource}`;
      }
    }
    case "MOVE_ROBBER": {
      const action = actionRecord[0] as MoveRobberAction;
      const tile = findTileByCoordinate(gameState, action[2][0]);
      const tileString = getTileString(tile);
      const robbedResource = actionRecord[1];
      const stolenResource = robbedResource ? ` (STOLE ${robbedResource})` : "";
      return `${player} ROBBED ${tileString}${stolenResource}`;
    }
    case "MARITIME_TRADE": {
      const label = humanizeTradeAction(action as MaritimeTradeAction);
      return `${player} TRADED ${label}`;
    }
    case "END_TURN":
      return `${player} ENDED TURN`;
    default:
      throw new Error(`Unknown action type: ${action[1]}`);
  }
}
export function humanizeTradeAction(action: MaritimeTradeAction): string {
  const out = action[2]
    .slice(0, 4)
    .filter((resource: unknown) => resource !== null);
  return `${out.length} ${out[0]} => ${action[2][4]}`;
}

export function findTileByCoordinate(gameState: GameState, coordinate: any) {
  for (const tile of Object.values(gameState.tiles)) {
    if (JSON.stringify(tile.coordinate) === JSON.stringify(coordinate)) {
      return tile;
    }
  }
  throw new Error(
    `Tile not found for coordinate: ${JSON.stringify(coordinate)}`
  );
}

export function getShortTileString(tile: Tile): string {
  return tile.type === "RESOURCE_TILE" ? tile.number.toString() : tile.type;
}

export function getTileString(tile: PlacedTile): string {
  const tileInfo = tile.tile;
  switch (tileInfo.type) {
    case "DESERT":
      return "THE DESERT";
    case "RESOURCE_TILE":
      return `${tileInfo.number} ${tileInfo.resource}`;
    default:
      throw new Error("getTileString() only works on Desert or Resource tiles");
  }
}

export function findTileById(gameState: GameState, tileId: number): PlacedTile {
  return gameState.tiles[tileId];
}
