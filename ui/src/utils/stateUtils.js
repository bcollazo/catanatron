export function isPlayersTurn(gameState) {
  return !gameState.bot_colors.includes(gameState.current_color);
}

export function playerKey(gameState, color) {
  return `P${gameState.colors.indexOf(color)}`;
}

export function getHumanColor(gameState) {
  return gameState.colors.find(
    (color) => !gameState.bot_colors.includes(color)
  );
}
