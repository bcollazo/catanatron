import React from "react";
import Loader from "react-loader-spinner";
import { HUMAN_COLOR } from "../constants";

import "./Prompt.scss";

export function humanizeAction(action) {
  switch (action[1]) {
    case "ROLL":
      return `CATANATRON ROLLED ${action[2]}`;
    case "DISCARD":
      return `CATANATRON DISCARDED ${action[2]}`;
    case "BUY_DEVELOPMENT_CARD":
      return `CATANATRON BOUGHT DEVELOPMENT CARD`;
    case "BUILD_FIRST_SETTLEMENT":
    case "BUILD_SECOND_SETTLEMENT":
    case "BUILD_SETTLEMENT":
    case "BUILD_CITY": {
      const parts = action[1].split("_");
      const building = parts[parts.length - 1];
      const tile = action[2];
      return `CATANATRON BUILT ${building} ON ${tile}`;
    }
    case "BUILD_INITIAL_ROAD": {
      const edge = action[2];
      return `CATANATRON BUILT INITIAL ROAD ON ${edge}`;
    }
    case "BUILD_ROAD": {
      const edge = action[2];
      return `CATANATRON BUILT ROAD ON ${edge}`;
    }
    case "PLAY_KNIGHT_CARD": {
      const tile = action[2];
      return `CATANATRON PLAYED KNIGHT CARD TO ${tile}`;
    }
    case "MOVE_ROBBER": {
      const tile = action[2];
      return `CATANATRON MOVED ROBBER TO ${tile}`;
    }

    case "END_TURN":
      return `CATANATRON ENDED TURN`;
    default:
      return `CATANATRON ${action.slice(1)}`;
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

export default function Prompt({ state, isBotThinking }) {
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
        <small>Computing</small>
      </>
    );
  } else if (state.winning_color) {
    prompt = `Game Over. Congrats, ${state.winning_color}!`;
  } else if (state.current_color === HUMAN_COLOR) {
    prompt = humanizePrompt(state.current_prompt);
  } else {
    // prompt = humanizeAction(state.actions[state.actions.length - 1]);
  }
  console.log(prompt);
  return <div className="prompt">{prompt}</div>;
}
