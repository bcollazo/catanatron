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
  | [DiscardGameAction, ResourceCard[]]
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
export type DiscardGameAction = [Color, "DISCARD", null];
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
  [ResourceCard] | [ResourceCard, ResourceCard]
];
export type MoveRobberAction = [
  Color,
  "MOVE_ROBBER",
  [TileCoordinate, string?]
];
export type MaritimeTradeAction = [
  Color,
  "MARITIME_TRADE",
  (ResourceCard | null)[]
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
  bot_labels?: Partial<Record<Color, string>>;
  colors: Color[];
  current_color: Color;
  winning_color?: Color;
  current_prompt: string;
  player_state: Record<string, PlayerState>;
  action_records: GameActionRecord[];
  policy_debug_records?: Array<{
    chosen_action_index: number;
    chosen_action_probability: number;
    /** Renormalized over legal actions only (recommended for reading the policy). */
    chosen_action_probability_given_valid?: number;
    chosen_action_description?: string;
    chosen_action_description_detailed?: string;
    state_value_estimate: number;
    top_actions: Array<{
      action_index: number;
      /** Raw softmax mass (often tiny except the peak). */
      probability: number;
      /** Share of mass among legal moves; sums to 1 over all legal indices. */
      probability_given_valid?: number;
      description: string;
      description_detailed: string;
    }>;
  } | null>;
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
  state_index: number;
};

export type ReplayCatalogItem = {
  game_id: string;
  state_index: number;
  turn_count: number;
  winner: Color | null;
  us_color: Color | null;
  went_first: boolean | null;
  won: boolean | null;
  us_final_vp: number | null;
  opp_final_vp: number | null;
  us_buy_dev: number;
  us_maritime_trades: number;
  us_build_city: number;
  us_build_settlement: number;
  us_play_knight: number;
  us_opening_pip_score: number | null;
  opp_opening_pip_score: number | null;
  opening_pip_diff: number | null;
  us_first_city_turn: number | null;
  us_action_build: number;
  us_action_trade: number;
  us_action_dev: number;
  us_action_robber: number;
  us_action_total: number;
  replay_source_folder: string | null;
  imported_at_utc: string | null;
};

export type PolicyActionAnalysis = {
  action_index: number;
  probability: number;
  description: string;
  description_detailed: string;
  next_state_value_estimate: number | null;
};

export type PolicyAnalysis = {
  success: true;
  state_index: number;
  model_path: string;
  top_n: number;
  state_value_estimate: number;
  top_actions: PolicyActionAnalysis[];
  chosen_action: {
    action_index: number;
    description: string;
    description_detailed: string;
  } | null;
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
