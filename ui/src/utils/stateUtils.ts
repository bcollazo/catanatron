type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";

type GameState = {
  bot_colors: Color[];
  colors: Color[];
  current_color: Color;
};

/**
 * Check if it's a human player's turn
 * @param gameState
 * @returns True if a human player needs to play
 */
export function isPlayersTurn(gameState: GameState) {
  return !gameState.bot_colors.includes(gameState.current_color);
}

export function playerKey(gameState: GameState, color: Color) {
  return `P${gameState.colors.indexOf(color)}`;
}

export function getHumanColor(gameState: GameState) {
  return gameState.colors.find(
    (color) => !gameState.bot_colors.includes(color)
  );
}
