export const RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"] as const;
export type Resource = (typeof RESOURCES)[number];
export type Color = "RED" | "BLUE" | "ORANGE" | "WHITE";
export type Coord = [number, number, number];
export type Action = [Color, string, any];
export type ActionRecord = [Action, any];
export interface Tile {
  coordinate: Coord;
  tile: {
    type: string;
    resource?: Resource | null;
    number?: number;
    direction?: string;
    id?: number;
  };
}
export interface Node {
  id: number;
  tile_coordinate: Coord;
  direction: string;
  color: Color | null;
  building: string | null;
}
export interface Edge {
  id: [number, number];
  tile_coordinate: Coord;
  direction: string;
  color: Color | null;
}
export interface GameState {
  settings?: {
    vps_to_win?: number;
    discard_limit?: number;
    friendly_robber?: boolean;
  };
  tiles: Tile[];
  nodes: Record<string, Node>;
  edges: Edge[];
  colors: Color[];
  bot_colors: Color[];
  player_state: Record<string, number | boolean>;
  action_records: ActionRecord[];
  state_index: number;
  current_color: Color;
  current_prompt: string;
  current_playable_actions: Action[];
  robber_coordinate: Coord;
  winning_color: Color | null;
  is_initial_build_phase: boolean;
  current_discard_count: number;
  longest_roads_by_player?: Partial<Record<Color, number>>;
}
export const playerColors: Record<Color, string> = {
  RED: "#ff7185",
  BLUE: "#62b2ff",
  ORANGE: "#ffb35e",
  WHITE: "#e2e9f4",
};
export const resourceColors: Record<Resource, string> = {
  WOOD: "#42d3a0",
  BRICK: "#fa906a",
  SHEEP: "#b6e478",
  WHEAT: "#f2ce68",
  ORE: "#a3a3a3",
};
export const title = (s: string) =>
  s
    .toLowerCase()
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
export const stat = (s: GameState, c: Color, name: string): number =>
  Number(s.player_state[`P${s.colors.indexOf(c)}_${name}`] ?? 0);
export const actionKey = (a: Action) => JSON.stringify(a);
export const same = (a: unknown, b: unknown) =>
  JSON.stringify(a) === JSON.stringify(b);
export function actionLabel(a: Action): string {
  if (a[1] === "MARITIME_TRADE") {
    const given = a[2].slice(0, -1).filter(Boolean);
    return `${given.length} ${title(given[0])} → 1 ${title(a[2].at(-1))}`;
  }
  if (a[1] === "MOVE_ROBBER")
    return a[2][1] ? `Steal from ${title(a[2][1])}` : "Move without stealing";
  if (a[1] === "PLAY_YEAR_OF_PLENTY") return a[2].map(title).join(" + ");
  if (a[1] === "PLAY_MONOPOLY" || a[1] === "DISCARD_RESOURCE")
    return title(a[2]);
  return title(a[1])
    .replace("Play ", "")
    .replace("Buy Development Card", "Development card");
}
export function recordLabel([a, result]: ActionRecord): string {
  if (a[1] === "ROLL")
    return `Rolled ${Array.isArray(result) ? result.join(" + ") + " = " + (result[0] + result[1]) : "the dice"}`;
  if (a[1] === "MOVE_ROBBER")
    return `${actionLabel(a)}${result ? ` · ${title(result)}` : ""}`;
  if (a[1] === "DISCARD_RESOURCE") return `Discarded ${title(a[2])}`;
  if (a[1] === "PLAY_MONOPOLY") return `Monopoly · ${title(a[2])}`;
  if (a[1] === "PLAY_YEAR_OF_PLENTY")
    return `Year of Plenty · ${actionLabel(a)}`;
  if (a[1] === "BUY_DEVELOPMENT_CARD")
    return `Bought ${result ? title(result) : "development card"}`;
  return actionLabel(a);
}
export function parseReplay(text: string): GameState[] {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(
      "This file is not valid JSON. Choose a Catanatron JSON export.",
    );
  }
  const raw = Array.isArray(value)
    ? value
    : value && typeof value === "object" && "states" in value
      ? (value as { states: unknown }).states
      : [value];
  if (!Array.isArray(raw) || !raw.length || !raw.every(isGameState))
    throw new Error(
      "Expected a Catanatron game snapshot or an array of snapshots. CSV statistics cannot reconstruct a board.",
    );
  return combineSnapshots(raw);
}
const object = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const coord = (v: unknown): v is Coord =>
  Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
const color = (v: unknown): v is Color =>
  typeof v === "string" && Object.hasOwn(playerColors, v);
const resource = (v: unknown): v is Resource =>
  typeof v === "string" && Object.hasOwn(resourceColors, v);
const nodeDirections = [
  "NORTH",
  "NORTHEAST",
  "SOUTHEAST",
  "SOUTH",
  "SOUTHWEST",
  "NORTHWEST",
];
function validAction(v: unknown): boolean {
  if (
    !Array.isArray(v) ||
    v.length !== 3 ||
    !color(v[0]) ||
    typeof v[1] !== "string"
  )
    return false;
  const payload = v[2];
  switch (v[1]) {
    case "BUILD_SETTLEMENT":
    case "BUILD_CITY":
      return Number.isInteger(payload);
    case "BUILD_ROAD":
      return (
        Array.isArray(payload) &&
        payload.length === 2 &&
        payload.every(Number.isInteger)
      );
    case "MOVE_ROBBER":
      return (
        Array.isArray(payload) &&
        coord(payload[0]) &&
        (payload[1] == null || color(payload[1]))
      );
    case "DISCARD_RESOURCE":
    case "PLAY_MONOPOLY":
      return resource(payload);
    case "PLAY_YEAR_OF_PLENTY":
      return (
        Array.isArray(payload) &&
        payload.length >= 1 &&
        payload.length <= 2 &&
        payload.every(resource)
      );
    case "MARITIME_TRADE":
      return (
        Array.isArray(payload) &&
        payload.length === 5 &&
        payload.every((p) => p === null || resource(p)) &&
        resource(payload[0]) &&
        resource(payload[4])
      );
    // Historic ROLL records include resolved dice in the action as well as result.
    case "ROLL":
      return (
        payload === null ||
        (Array.isArray(payload) &&
          payload.length === 2 &&
          payload.every(Number.isInteger))
      );
    case "BUY_DEVELOPMENT_CARD":
      return (
        payload === null ||
        [
          "KNIGHT",
          "MONOPOLY",
          "YEAR_OF_PLENTY",
          "ROAD_BUILDING",
          "VICTORY_POINT",
        ].includes(payload)
      );
    case "END_TURN":
    case "PLAY_KNIGHT_CARD":
    case "PLAY_ROAD_BUILDING":
      return payload === null;
    // The engine also serializes experimental domestic trade records.
    case "OFFER_TRADE":
    case "ACCEPT_TRADE":
    case "REJECT_TRADE":
    case "CONFIRM_TRADE":
    case "CANCEL_TRADE":
      return payload === null || Array.isArray(payload);
    default:
      return false;
  }
}
function isGameState(s: unknown): s is GameState {
  if (!object(s)) return false;
  return (
    Array.isArray(s.tiles) &&
    s.tiles.length > 0 &&
    s.tiles.every(
      (t) =>
        object(t) &&
        coord(t.coordinate) &&
        object(t.tile) &&
        ["WATER", "PORT", "DESERT", "RESOURCE_TILE"].includes(
          String(t.tile.type),
        ) &&
        (t.tile.resource == null || resource(t.tile.resource)) &&
        (t.tile.type !== "RESOURCE_TILE" ||
          (resource(t.tile.resource) &&
            Number.isInteger(t.tile.number) &&
            Number(t.tile.number) >= 2 &&
            Number(t.tile.number) <= 12 &&
            t.tile.number !== 7)),
    ) &&
    object(s.nodes) &&
    Object.values(s.nodes).every(
      (n) =>
        object(n) &&
        Number.isInteger(n.id) &&
        coord(n.tile_coordinate) &&
        nodeDirections.includes(String(n.direction)) &&
        (n.color === null || color(n.color)) &&
        (n.building === null ||
          n.building === "SETTLEMENT" ||
          n.building === "CITY"),
    ) &&
    Array.isArray(s.edges) &&
    s.edges.every(
      (e) =>
        object(e) &&
        Array.isArray(e.id) &&
        e.id.length === 2 &&
        e.id.every(Number.isInteger) &&
        coord(e.tile_coordinate) &&
        (e.color === null || color(e.color)),
    ) &&
    Array.isArray(s.colors) &&
    s.colors.length >= 2 &&
    s.colors.length <= 4 &&
    s.colors.every(color) &&
    Array.isArray(s.bot_colors) &&
    s.bot_colors.every(color) &&
    color(s.current_color) &&
    (s.winning_color === null || color(s.winning_color)) &&
    typeof s.is_initial_build_phase === "boolean" &&
    Number.isInteger(s.current_discard_count) &&
    typeof s.current_prompt === "string" &&
    object(s.player_state) &&
    Object.values(s.player_state).every(
      (v) => typeof v === "boolean" || typeof v === "number",
    ) &&
    Array.isArray(s.action_records) &&
    s.action_records.every(
      (r) =>
        Array.isArray(r) &&
        r.length === 2 &&
        validAction(r[0]) &&
        (r[1] === null ||
          typeof r[1] === "string" ||
          (Array.isArray(r[1]) && r[1].every(Number.isFinite))),
    ) &&
    Array.isArray(s.current_playable_actions) &&
    s.current_playable_actions.every(validAction) &&
    coord(s.robber_coordinate) &&
    Number.isInteger(s.state_index) &&
    Number(s.state_index) >= 0
  );
}
export function combineSnapshots(snapshots: GameState[]): GameState[] {
  const sorted = [...snapshots].sort((a, b) => a.state_index - b.state_index);
  if (new Set(sorted.map((s) => s.state_index)).size !== sorted.length)
    throw new Error("Choose snapshots from one game with unique move numbers.");
  const first = sorted[0];
  if (
    sorted.some(
      (s, i) =>
        !same(s.tiles, first.tiles) ||
        !same(s.colors, first.colors) ||
        (i > 0 &&
          !same(
            s.action_records.slice(0, sorted[i - 1].action_records.length),
            sorted[i - 1].action_records,
          )),
    )
  )
    throw new Error(
      "These snapshots belong to different games. Import one game at a time.",
    );
  return sorted;
}
export const center = ([q, , r]: Coord): [number, number] => [
  Math.sqrt(3) * 58 * (q + r / 2),
  87 * r,
];
export function nodePoint(n: Node): [number, number] {
  const angle =
    ([
      "NORTH",
      "NORTHEAST",
      "SOUTHEAST",
      "SOUTH",
      "SOUTHWEST",
      "NORTHWEST",
    ].indexOf(n.direction) *
      Math.PI) /
      3 -
    Math.PI / 2;
  const [x, y] = center(n.tile_coordinate);
  return [x + 58 * Math.cos(angle), y + 58 * Math.sin(angle)];
}

// Same coastal corners as the engine's PORT_DIRECTION_TO_NODEREFS. Node IDs in
// GameEncoder may use another tile as their origin, so match their world positions.
const portCorners: Record<string, [string, string]> = {
  WEST: ["NORTHWEST", "SOUTHWEST"],
  NORTHWEST: ["NORTH", "NORTHWEST"],
  NORTHEAST: ["NORTHEAST", "NORTH"],
  EAST: ["SOUTHEAST", "NORTHEAST"],
  SOUTHEAST: ["SOUTH", "SOUTHEAST"],
  SOUTHWEST: ["SOUTHWEST", "SOUTH"],
};
export function portGeometry(game: GameState, port: Tile) {
  const corners = portCorners[port.tile.direction ?? ""];
  if (!corners) return null;
  const nodes = corners.map((direction) => {
    const p = nodePoint({
      id: -1,
      tile_coordinate: port.coordinate,
      direction,
      color: null,
      building: null,
    });
    return Object.values(game.nodes).find((n) => {
      const q = nodePoint(n);
      return Math.hypot(p[0] - q[0], p[1] - q[1]) < 0.001;
    });
  });
  if (!nodes[0] || !nodes[1]) return null;
  const ends = [nodePoint(nodes[0]), nodePoint(nodes[1])] as const;
  const midpoint = [
    (ends[0][0] + ends[1][0]) / 2,
    (ends[0][1] + ends[1][1]) / 2,
  ];
  const [cx, cy] = center(port.coordinate);
  const distance = Math.hypot(cx - midpoint[0], cy - midpoint[1]);
  const badge: [number, number] = [
    midpoint[0] + ((cx - midpoint[0]) * 26) / distance,
    midpoint[1] + ((cy - midpoint[1]) * 26) / distance,
  ];
  return { nodes: [nodes[0].id, nodes[1].id], ends, badge };
}
