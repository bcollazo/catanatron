import { HUMAN_COLOR } from "../constants";

export function isPlayersTurn(gameState) {
  return gameState.current_color === HUMAN_COLOR;
}
