import React from "react";
import Loader from "react-loader-spinner";
import { HUMAN_COLOR } from "../constants";

import "./Prompt.scss";

export function humanizeAction(action) {
  const player = action[0] === HUMAN_COLOR ? "YOU" : "BOT";
  switch (action[1]) {
    case "ROLL":
      return `${player} ROLLED A ${action[2][0] + action[2][1]}`;
    case "DISCARD":
      return `${player} DISCARDED ${action[2]}`;
    case "BUY_DEVELOPMENT_CARD":
      return `${player} BOUGHT DEVELOPMENT CARD`;
    case "BUILD_FIRST_SETTLEMENT":
    case "BUILD_SECOND_SETTLEMENT":
    case "BUILD_SETTLEMENT":
    case "BUILD_CITY": {
      const parts = action[1].split("_");
      const building = parts[parts.length - 1];
      const tile = action[2];
      return `${player} BUILT ${building} ON ${tile}`;
    }
    case "BUILD_INITIAL_ROAD":
    case "BUILD_ROAD": {
      const edge = action[2];
      return `${player} BUILT ROAD ON ${edge}`;
    }
    case "PLAY_KNIGHT_CARD": {
      const tile = action[2];
      return `${player} PLAYED KNIGHT CARD TO ${tile}`;
    }
    case "MOVE_ROBBER": {
      const tile = action[2];
      return `${player} ROBBED ${tile}`;
    }
    case "END_TURN":
      return `${player} ENDED TURN`;
    default:
      return `${player} ${action.slice(1)}`;
  }
}

function humanizePrompt(current_prompt) {
  switch (current_prompt) {
    case "ROLL":
      return `YOUR TURN`;
    case "PLAY_TURN":
      return `YOUR TURN`;
    case "BUILD_FIRST_SETTLEMENT":
    case "BUILD_SECOND_SETTLEMENT":
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
    prompt = (
      <>
        <Loader
          className="loader"
          type="Grid"
          color="#ffffff"
          height={10}
          width={10}
        />
        <small>Bots Turn</small>
      </>
    );
  } else if (gameState.winning_color) {
    prompt = `Game Over. Congrats, ${gameState.winning_color}!`;
  } else if (gameState.current_color === HUMAN_COLOR) {
    prompt = humanizePrompt(gameState.current_prompt);
  } else {
    // prompt = humanizeAction(gameState.actions[gameState.actions.length - 1]);
  }
  return <div className="prompt">{prompt}</div>;
}
