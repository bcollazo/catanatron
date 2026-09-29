export type Card = ResourceCard | DevelopmentCard | VictoryPointCard;
export type DevelopmentCard =
  | "KNIGHT"
  | "MONOPOLY"
  | "YEAR_OF_PLENTY"
  | "ROAD_BUILDING";

export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";
export type TileCoordinate = [number, number, number];

export type GameActionRecord =
  // These are the special cases
  | [RollGameAction, [number, number]]
  | [DiscardGameAction, ResourceCard]
  | [MoveRobberAction, ResourceCard | null]
  | [BuyDevelopmentCardAction, DevelopmentCard]
  // These are deterministic and carry no extra info
  | [BuildSettlementAction, null]
  | [BuildCityAction, null]
  | [BuildRoadAction, null]
  | [PlayKnightCardAction, null]
  | [PlayRoadBuildingAction, null]
  | [PlayMonopolyAction, null]
  | [PlayYearOfPlentyAction, null]
  | [MaritimeTradeAction, null]
  | [EndTurnAction, null];

export type RollGameAction = [Color, "ROLL", null];
export type DiscardGameAction = [Color, "DISCARD_RESOURCE", ResourceCard];
export type BuyDevelopmentCardAction = [Color, "BUY_DEVELOPMENT_CARD", null];
export type BuildSettlementAction = [Color, "BUILD_SETTLEMENT", number];
export type BuildCityAction = [Color, "BUILD_CITY", number];
export type BuildRoadAction = [Color, "BUILD_ROAD", [number, number]];
export type PlayKnightCardAction = [Color, "PLAY_KNIGHT_CARD", null];
export type PlayRoadBuildingAction = [Color, "PLAY_ROAD_BUILDING", null];
export type PlayMonopolyAction = [Color, "PLAY_MONOPOLY", ResourceCard];
export type PlayYearOfPlentyAction = [
  Color,
  "PLAY_YEAR_OF_PLENTY",
  [ResourceCard] | [ResourceCard, ResourceCard],
];
export type MoveRobberAction = [
  Color,
  "MOVE_ROBBER",
  [TileCoordinate, string?],
];
export type MaritimeTradeAction = [
  Color,
  "MARITIME_TRADE",
  (ResourceCard | null)[],
];
export type EndTurnAction = [Color, "END_TURN", null];

export type GameAction =
  | RollGameAction
  | DiscardGameAction
  | BuyDevelopmentCardAction
  | BuildSettlementAction
  | BuildCityAction
  | BuildRoadAction
  | PlayKnightCardAction
  | PlayRoadBuildingAction
  | PlayMonopolyAction
  | PlayYearOfPlentyAction
  | MoveRobberAction
  | MaritimeTradeAction
  | EndTurnAction;

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

type WaterTile = {
  type: "WATER";
};

type PortTile = {
  id: number;
  type: "PORT";
  direction: Direction;
  resource: ResourceCard | null;
};

export type Tile = ResourceTile | DesertTile | WaterTile | PortTile;

export type PlacedTile = {
  coordinate: TileCoordinate;
  tile: Tile;
};

export type GameNode = {
  id: number;
  tile_coordinate: TileCoordinate;
  direction: Direction;
  building: Building | null;
  color: Color | null;
};

export type GameEdge = {
  id: [number, number];
  color: Color | null;
  direction: Direction;
  tile_coordinate: TileCoordinate;
};

/**
 * The single payload GET /api/games/.../states/... returns: the game's state
 * document, plus the few things a browser needs that are not state — the
 * legal actions, which seats are bots, and the board geometry.
 * Built by catanatron.serialization.web_view.
 */
export type GameState = {
  tiles: PlacedTile[];
  adjacent_tiles: Record<string, Tile[]>;
  bot_colors: Color[];
  colors: Color[];
  current_color: Color;
  winning_color: Color | null;
  current_prompt: string;
  player_state: Record<string, PlayerState>;
  action_records: GameActionRecord[];
  robber_coordinate: TileCoordinate;
  current_discard_count: number;
  nodes: Record<string, GameNode>;
  edges: GameEdge[];
  current_playable_actions: GameAction[];
  is_initial_build_phase: boolean;
  longest_roads_by_player: Partial<Record<Color, number>>;
  edgeActions?: GameAction[];
  nodeActions?: GameAction[];
  state_index: number;

  // Also present, from the state document itself.
  schema_version: number;
  game: {
    id: string;
    vps_to_win: number;
    discard_limit: number;
    friendly_robber: boolean;
  };
  num_turns: number;
  current_player_index: number;
  current_turn_index: number;
  resource_freqdeck: number[];
  development_listdeck: Record<string, number>;
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
