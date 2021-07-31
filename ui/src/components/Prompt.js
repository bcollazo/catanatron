import React from "react";
import { isPlayersTurn } from "../utils/stateUtils";

import "./Prompt.scss";

export function humanizeAction(action, botColors) {
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
      const tile = action[2];
      return `${player} BUILT ${building} ON ${tile}`;
    }
    case "BUILD_ROAD": {
      const edge = action[2];
      return `${player} BUILT ROAD ON ${edge}`;
    }
    case "PLAY_KNIGHT_CARD": {
      return `${player} PLAYED KNIGHT CARD`;
    }
    case "PLAY_YEAR_OF_PLENTY": {
      return `${player} YEAR OF PLENTY ${action[2]}`;
    }
    case "MOVE_ROBBER": {
      const tile = action[2];
      return `${player} ROBBED ${tile}`;
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
