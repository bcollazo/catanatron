export type Card = ResourceCard | DevelopmentCard | VictoryPointCard;
export type DevelopmentCard =
  | "KNIGHT"
  | "MONOPOLY"
  | "YEAR_OF_PLENTY"
  | "ROAD_BUILDING";

export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";
export type TileCoordinate = [number, number, number];

export type GameAction =
  | [Color, "ROLL", [number, number] | null]
  | [Color, "DISCARD", null]
  | [Color, "BUY_DEVELOPMENT_CARD", null]
  | [Color, "BUILD_SETTLEMENT", number]
  | [Color, "BUILD_CITY", number]
  | [Color, "BUILD_ROAD", [number, number]]
  | [Color, "PLAY_KNIGHT_CARD", null]
  | [Color, "PLAY_ROAD_BUILDING", null]
  | [Color, "PLAY_MONOPOLY", ResourceCard]
  | [
      Color,
      "PLAY_YEAR_OF_PLENTY",
      [ResourceCard] | [ResourceCard, ResourceCard]
    ]
  | [Color, "MOVE_ROBBER", [TileCoordinate, string?, string?]]
  | [Color, "MARITIME_TRADE", any]
  | [Color, "END_TURN", null];

export type PlayerState = any;
export type VictoryPointCard = "VICTORY_POINT";
export type ResourceCard = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";
export type Building = "SETTLEMENT" | "CITY";

type ResourceTile = {
  id: number;
  type: "RESOURCE_TILE";
  resource: ResourceCard;
  number: number;
};

type DesertTile = {
  id: number;
  type: "DESERT";
};

type PortTile = {
  id: number;
  type: "PORT";
  direction: Direction;
  resource: ResourceCard;
};

export type Tile = ResourceTile | DesertTile | PortTile;

export type PlacedTile = {
  coordinate: TileCoordinate;
  tile: Tile;
};

export type GameState = {
  tiles: PlacedTile[];
  adjacent_tiles: Record<string, Tile[]>;
  bot_colors: Color[];
  colors: Color[];
  current_color: Color;
  winning_color?: Color;
  current_prompt: string;
  player_state: Record<string, PlayerState>;
  actions: GameAction[];
  robber_coordinate: TileCoordinate;
  nodes: Array<{
    id: number;
    tile_coordinate: TileCoordinate;
    direction: Direction;
    building: Building | null;
    color: Color | null;
  }>;
  edges: Array<{
    id: [number, number];
    color: Color | null;
    direction: Direction;
    tile_coordinate: TileCoordinate;
  }>;
  current_playable_actions: GameAction[];
  is_initial_build_phase: boolean;
  edgeActions?: GameAction[];
  nodeActions?: GameAction[];
};
const DIRECTIONS = [
  "NORTH",
  "NORTHEAST",
  "SOUTHEAST",
  "SOUTH",
  "SOUTHWEST",
  "NORTHWEST",
  "EAST",
  "WEST",
] as const;

export type Direction = (typeof DIRECTIONS)[number];
