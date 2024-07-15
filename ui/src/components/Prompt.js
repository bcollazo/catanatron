import React from "react";
import { isPlayersTurn } from "../utils/stateUtils";

import "./Prompt.scss";

function findTileByCoordinate(gameState, coordinate) {
  for (const tile of Object.values(gameState.tiles)) {
    if (JSON.stringify(tile.coordinate) === JSON.stringify(coordinate)) {
      return tile;
    }
  }
}

function findTileById(gameState, tileId) {
  return gameState.tiles[tileId];
}

function getTileString(tile) {
  const { number = "THE", resource = "DESERT" } = tile.tile;
  return `${number} ${resource}`;
}

function getShortTileString(tileTile) {
  return tileTile.number || tileTile.type;
}

export function humanizeAction(gameState, action) {
  const botColors = gameState.bot_colors;
  const player = botColors.includes(action[0]) ? "BOT" : "YOU";
  switch (action[1]) {
    case "ROLL":
      return `${player} ROLLED A ${action[2][0] + action[2][1]}`;
    case "DISCARD":
      return `${player} DISCARDED`;
    case "BUY_DEVELOPMENT_CARD":
      return `${player} BOUGHT DEVELOPMENT CARD`;
    case "BUILD_SETTLEMENT":
    case "BUILD_CITY": {
      const parts = action[1].split("_");
      const building = parts[parts.length - 1];
      const tileId = action[2];
      const tiles = gameState.adjacent_tiles[tileId];
      const tileString = tiles.map(getShortTileString).join("-");
      return `${player} BUILT ${building} ON ${tileString}`;
    }
    case "BUILD_ROAD": {
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
      return `${player} PLAYED ROAD BUILDING`
    }
    case "PLAY_MONOPOLY": {
      return `${player} MONOPOLIZED ${action[2]}`;
    }
    case "PLAY_YEAR_OF_PLENTY": {
      const firstResource = action[2][0];
      const secondResource = action[2][1];
      if (secondResource) {
        return `${player} PLAYED YEAR OF PLENTY. CLAIMED ${firstResource} AND ${secondResource}`;
      } else {
        return `${player} PLAYED YEAR OF PLENTY. CLAIMED ${firstResource}`;
      }
    }
    case "MOVE_ROBBER": {
      const tile = findTileByCoordinate(gameState, action[2][0]);
      const tileString = getTileString(tile);
      const stolenResource = action[2][2] ? ` (STOLE ${action[2][2]})` : '';
      return `${player} ROBBED ${tileString}${stolenResource}`;
    }
    case "MARITIME_TRADE": {
      const label = humanizeTradeAction(action);
      return `${player} TRADED ${label}`;
    }
    case "END_TURN":
      return `${player} ENDED TURN`;
    default:
      return `${player} ${action.slice(1)}`;
  }
}

export function humanizeTradeAction(action) {
  const out = action[2].slice(0, 4).filter((resource) => resource !== null);
  return `${out.length} ${out[0]} => ${action[2][4]}`;
}

function humanizePrompt(current_prompt) {
  switch (current_prompt) {
    case "ROLL":
      return `YOUR TURN`;
    case "PLAY_TURN":
      return `YOUR TURN`;
    case "BUILD_INITIAL_SETTLEMENT":
    case "BUILD_INITIAL_ROAD":
    default: {
      const prompt = current_prompt.replaceAll("_", " ");
      return `PLEASE ${prompt}`;
    }
  }
}

export default function Prompt({ gameState, isBotThinking }) {
  let prompt = "";
  if (isBotThinking) {
    // Do nothing, but still render.
  } else if (gameState.winning_color) {
    prompt = `Game Over. Congrats, ${gameState.winning_color}!`;
  } else if (isPlayersTurn(gameState)) {
    prompt = humanizePrompt(gameState.current_prompt);
  } else {
    // prompt = humanizeAction(gameState.actions[gameState.actions.length - 1], gameState.bot_colors);
  }
  return <div className="prompt">{prompt}</div>;
}
