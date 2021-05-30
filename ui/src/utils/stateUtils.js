import { HUMAN_COLOR } from "../constants";

export function isPlayersTurn(gameState) {
  return gameState.current_color === HUMAN_COLOR;
}

export function playerKey(gameState, color) {
  return `P${gameState.colors.indexOf(color)}`;
}
