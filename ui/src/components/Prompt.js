import React from "react";

import "./Prompt.scss";

function humanizeAction(action) {
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

function humanizePrompt(state) {
  switch (state.current_prompt) {
    case "ROLL":
      return `YOUR TURN`;
    case "PLAY_TURN":
      return `YOUR TURN`;
    case "BUILD_FIRST_SETTLEMENT":
    case "BUILD_SECOND_SETTLEMENT":
    case "BUILD_INITIAL_ROAD":
    default: {
      const prompt = state.current_prompt.replaceAll("_", " ");
      return `PLEASE ${prompt}`;
    }
  }
}

export default function Prompt({ actionQueue, state }) {
  let prompt = "";
  if (state.winning_color) {
    prompt = `Game Over. Congrats, ${state.winning_color}!`;
  } else if (actionQueue.length === 0) {
    prompt = humanizePrompt(state);
  } else {
    prompt = humanizeAction(actionQueue[0]);
  }
  return <div className="prompt">{state && prompt}</div>;
}
