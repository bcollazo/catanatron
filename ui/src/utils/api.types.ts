
export type Card = ResourceCard | DevelopmentCard | VictoryPointCard;
export type DevelopmentCard = "KNIGHT" |
  "MONOPOLY" |
  "YEAR_OF_PLENTY" |
    "ROAD_BUILDING";

export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";

export type GameAction = [Color, "ROLL", [number, number]] |
[Color, "DISCARD"] |
[Color, "BUY_DEVELOPMENT_CARD"] |
[Color, "BUILD_SETTLEMENT", number] |
[Color, "BUILD_CITY", number] |
[Color, "BUILD_ROAD", [number, number]] |
[Color, "PLAY_KNIGHT_CARD"] |
[Color, "PLAY_ROAD_BUILDING"] |
[Color, "PLAY_MONOPOLY", string] |
[Color, "PLAY_YEAR_OF_PLENTY", [string, string?]] |
[Color, "MOVE_ROBBER", [[number, number], string?]] |
[Color, "MARITIME_TRADE", any] |
[Color, "END_TURN"]; // TODO - fix types

export type Tile = { number: string; resource: string; type: string; };
export type PlacedTile = {
  id: string;
  coordinate: any; // Replace with actual type if known
  tile: Tile;
};

export type PlayerState = any;
export type VictoryPointCard = "VICTORY_POINT";
export type ResourceCard = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

export type GameState = {
  tiles: Record<string, PlacedTile>;
  adjacent_tiles: Record<string, Tile[]>;
  bot_colors: Color[];
  colors: Color[];
  current_color: Color;
  winning_color?: Color;
  current_prompt: string;
  player_state: Record<string, PlayerState>;
  actions: GameAction[];
};


