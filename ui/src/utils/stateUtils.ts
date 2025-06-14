export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";

export type GameAction =
  | [Color, "ROLL", [number, number]]
  | [Color, "DISCARD"]
  | [Color, "BUY_DEVELOPMENT_CARD"]
  | [Color, "BUILD_SETTLEMENT", number]
  | [Color, "BUILD_CITY", number]
  | [Color, "BUILD_ROAD", [number, number]]
  | [Color, "PLAY_KNIGHT_CARD"]
  | [Color, "PLAY_ROAD_BUILDING"]
  | [Color, "PLAY_MONOPOLY", string]
  | [Color, "PLAY_YEAR_OF_PLENTY", [string, string?]]
  | [Color, "MOVE_ROBBER", [[number, number], string?]]
  | [Color, "MARITIME_TRADE", any]
  | [Color, "END_TURN"]; // TODO - fix types

export type Tile = { number: string; resource: string; type: string };
export type PlacedTile = {
  id: string;
  coordinate: any; // Replace with actual type if known
  tile: Tile;
};

export type GameState = {
  tiles: Record<string, PlacedTile>;
  adjacent_tiles: Record<string, Tile[]>;
  bot_colors: Color[];
  colors: Color[];
  current_color: Color;
  winning_color?: Color;
  current_prompt: string;
};

/**
 * Check if it's a human player's turn
 * @param gameState
 * @returns True if a human player needs to play
 */
export function isPlayersTurn(gameState: GameState): boolean {
  return !gameState.bot_colors.includes(gameState.current_color);
}

export function playerKey(gameState: GameState, color: Color): string {
  return `P${gameState.colors.indexOf(color)}`;
}

export function getHumanColor(gameState: GameState): Color | undefined {
  return gameState.colors.find(
    (color) => !gameState.bot_colors.includes(color)
  );
}
