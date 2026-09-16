import { useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { Minus, Plus, Scan, Home, Building2, ShieldAlert } from "lucide-react";
import {
  center,
  portGeometry,
  nodePoint,
  same,
  resourceColors,
  playerColors,
  title,
  type Action,
  type GameState,
} from "./model";
import { ResourceIcon } from "./Resources";
import { Port } from "./Port";

const hex = (radius: number) =>
  Array.from({ length: 6 }, (_, i) => {
    const a = (i * Math.PI) / 3 - Math.PI / 2;
    return `${Math.cos(a) * radius},${Math.sin(a) * radius}`;
  }).join(" ");
function activate(event: React.KeyboardEvent, callback: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    callback();
  }
}

export function Board({
  game,
  actions = [],
  onAction,
  onChoices,
  animated = true,
}: {
  game: GameState;
  actions?: Action[];
  onAction?: (a: Action) => void;
  onChoices?: (a: Action[]) => void;
  animated?: boolean;
}) {
  const reduced = useReducedMotion();
  const [info, setInfo] = useState("");
  const points = game.tiles.flatMap((t) => {
    if (t.tile.type === "WATER") return [];
    const p =
      t.tile.type === "PORT"
        ? portGeometry(game, t)?.badge
        : center(t.coordinate);
    if (!p) return [];
    const padding = t.tile.type === "PORT" ? 20 : 62;
    return [
      [p[0] - padding, p[1] - padding],
      [p[0] + padding, p[1] + padding],
    ];
  });
  const minX = Math.min(...points.map((p) => p[0])),
    minY = Math.min(...points.map((p) => p[1]));
  const width = Math.max(...points.map((p) => p[0])) - minX,
    height = Math.max(...points.map((p) => p[1])) - minY;
  const last = game.action_records.at(-1);
  const roll =
    last?.[0][1] === "ROLL" && Array.isArray(last[1])
      ? last[1][0] + last[1][1]
      : 0;
  const move = (options: Action[]) =>
    options.length === 1 ? onAction?.(options[0]) : onChoices?.(options);
  const nodeMap = Object.fromEntries(
    Object.values(game.nodes).map((n) => [n.id, nodePoint(n)]),
  );
  const [rx, ry] = center(game.robber_coordinate);
  return (
    <div className="board-shell">
      <TransformWrapper
        minScale={0.75}
        maxScale={3}
        initialScale={1}
        wheel={{ step: 0.12 }}
        doubleClick={{ disabled: true }}
        panning={{ excluded: ["board-target"] }}
      >
        {({ zoomIn, zoomOut, resetTransform }) => (
          <>
            <div className="board-controls">
              <button aria-label="Zoom in" onClick={() => zoomIn()}>
                <Plus size={18} />
              </button>
              <button aria-label="Zoom out" onClick={() => zoomOut()}>
                <Minus size={18} />
              </button>
              <button aria-label="Fit board" onClick={() => resetTransform()}>
                <Scan size={18} />
              </button>
            </div>
            <TransformComponent
              wrapperClass="board-transform"
              contentClass="board-content"
            >
              <svg
                className="game-board"
                viewBox={`${minX} ${minY} ${width} ${height}`}
                role="group"
                aria-label="Catan game board"
              >
                <defs>
                  <pattern
                    id="circuit-grid"
                    width="20"
                    height="20"
                    patternUnits="userSpaceOnUse"
                  >
                    <circle cx="1" cy="1" r=".7" fill="#667d99" opacity=".3" />
                  </pattern>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="3" />
                  </filter>
                </defs>
                <rect
                  x={minX}
                  y={minY}
                  width={width}
                  height={height}
                  fill="url(#circuit-grid)"
                />
                {game.tiles.map(({ coordinate, tile }) => {
                  if (tile.type === "WATER") return null;
                  if (tile.type === "PORT")
                    return (
                      <Port
                        key={coordinate.join(",")}
                        game={game}
                        port={{ coordinate, tile }}
                        onInspect={setInfo}
                      />
                    );
                  const [x, y] = center(coordinate);
                  const options = actions.filter(
                    (a) => a[1] === "MOVE_ROBBER" && same(a[2][0], coordinate),
                  );
                  const resource = tile.resource,
                    color = resource ? resourceColors[resource] : "#c29a63",
                    active = options.length > 0;
                  const description = `${resource ? title(resource) : "Desert"}${tile.number ? ` · rolls ${tile.number}` : ""}`;
                  const inspect = () =>
                    active ? move(options) : setInfo(description);
                  return (
                    <g
                      key={coordinate.join(",")}
                      transform={`translate(${x} ${y})`}
                      className="land-tile"
                      data-tile-coordinate={coordinate.join(",")}
                      data-resource={resource ?? "DESERT"}
                    >
                      <g
                        className={
                          active ? "board-target tile-target" : "tile-info"
                        }
                        role="button"
                        tabIndex={0}
                        aria-label={
                          active ? `Move robber to ${description}` : description
                        }
                        onClick={inspect}
                        onKeyDown={(e) => activate(e, inspect)}
                      >
                        <polygon
                          points={hex(54)}
                          fill={color}
                          fillOpacity={resource ? 0.13 : 0.42}
                          stroke={active ? "var(--accent)" : color}
                          strokeOpacity={active ? 1 : 0.38}
                          strokeWidth={active ? 2.5 : 1}
                        />
                        <polygon
                          points={hex(48)}
                          fill="none"
                          stroke={color}
                          strokeOpacity=".1"
                        />
                        {resource && (
                          <g
                            transform={
                              tile.number
                                ? "translate(-17 -34)"
                                : "translate(-17 -17)"
                            }
                            color={color}
                          >
                            <ResourceIcon resource={resource} size={34} />
                          </g>
                        )}
                        {tile.number && (
                          <>
                            <circle
                              cy="22"
                              r="18"
                              fill="#080b10"
                              stroke={color}
                              strokeOpacity=".45"
                            />
                            <text
                              y="24"
                              className={`tile-number ${tile.number === 6 || tile.number === 8 ? "hot" : ""}`}
                            >
                              {tile.number}
                            </text>
                            <text
                              y="31"
                              className={`tile-pips ${tile.number === 6 || tile.number === 8 ? "hot" : ""}`}
                            >
                              {"•".repeat(6 - Math.abs(7 - tile.number))}
                            </text>
                          </>
                        )}
                        <title>{description}</title>
                      </g>
                      {animated &&
                        !reduced &&
                        roll === tile.number &&
                        !same(coordinate, game.robber_coordinate) && (
                          <motion.polygon
                            key={game.state_index}
                            points={hex(52)}
                            fill="none"
                            stroke={color}
                            initial={{ opacity: 1, scale: 0.8 }}
                            animate={{ opacity: 0, scale: 1.2 }}
                            transition={{ duration: 1.2 }}
                          />
                        )}
                    </g>
                  );
                })}
                {game.edges.map((edge) => {
                  const p = nodeMap[edge.id[0]],
                    q = nodeMap[edge.id[1]];
                  if (!p || !q) return null;
                  const a = actions.find(
                    (a) =>
                      a[1] === "BUILD_ROAD" &&
                      same(
                        [...a[2]].sort((a, b) => a - b),
                        [...edge.id].sort((a, b) => a - b),
                      ),
                  );
                  return (
                    <g key={edge.id.join(",")}>
                      {edge.color && (
                        <motion.path
                          key={edge.color}
                          d={`M${p} L${q}`}
                          stroke={playerColors[edge.color]}
                          strokeWidth="7"
                          strokeLinecap="round"
                          initial={
                            animated ? { pathLength: 0, opacity: 0 } : false
                          }
                          animate={{ pathLength: 1, opacity: 1 }}
                          transition={{ duration: reduced ? 0 : 0.4 }}
                        />
                      )}
                      {a && (
                        <g
                          className="board-target"
                          role="button"
                          tabIndex={0}
                          aria-label={`Build road ${edge.id.join(" to ")}`}
                          onClick={() => onAction?.(a)}
                          onKeyDown={(e) => activate(e, () => onAction?.(a))}
                        >
                          <path
                            d={`M${p} L${q}`}
                            stroke="transparent"
                            strokeWidth="24"
                          />
                          <path
                            className="legal-road"
                            d={`M${p} L${q}`}
                            stroke="var(--accent)"
                            strokeWidth="5"
                            strokeDasharray="3 7"
                            strokeLinecap="round"
                          />
                        </g>
                      )}
                    </g>
                  );
                })}
                {Object.values(game.nodes).map((n) => {
                  const [x, y] = nodePoint(n);
                  const a = actions.find(
                    (a) =>
                      (a[1] === "BUILD_SETTLEMENT" || a[1] === "BUILD_CITY") &&
                      a[2] === n.id,
                  );
                  const Icon = n.building === "CITY" ? Building2 : Home;
                  return (
                    <g key={n.id} transform={`translate(${x} ${y})`}>
                      {n.building && n.color && (
                        <motion.g
                          key={n.building + n.color}
                          initial={animated ? { scale: 0, opacity: 0 } : false}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{
                            type: "spring",
                            stiffness: 280,
                            damping: 18,
                          }}
                        >
                          <circle
                            r="12"
                            fill="#09101a"
                            stroke={playerColors[n.color]}
                            strokeWidth="1.5"
                          />
                          <Icon
                            x={-9}
                            y={-9}
                            size={18}
                            color={playerColors[n.color]}
                          />
                          <title>
                            {title(n.color)} {title(n.building)}
                          </title>
                        </motion.g>
                      )}
                      {a && (
                        <g
                          className="board-target"
                          role="button"
                          tabIndex={0}
                          aria-label={`${title(a[1])} at node ${n.id}`}
                          onClick={() => onAction?.(a)}
                          onKeyDown={(e) => activate(e, () => onAction?.(a))}
                        >
                          <circle r="17" fill="transparent" />
                          <circle
                            className="legal-node"
                            r="10"
                            fill="#142a47"
                            stroke="var(--accent)"
                            strokeWidth="2"
                          />
                          <text
                            textAnchor="middle"
                            y="5"
                            fontSize="18"
                            fill="#d8ebff"
                          >
                            +
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
                <motion.g
                  className="robber"
                  initial={false}
                  animate={{ x: rx + 29, y: ry - 25 }}
                  transition={{ type: "spring", stiffness: 100, damping: 18 }}
                  pointerEvents="none"
                >
                  <circle r="15" fill="#111a28" stroke="#f2edf6" />
                  <ShieldAlert x={-10} y={-10} size={20} color="#f2edf6" />
                  <title>Robber · blocks production</title>
                </motion.g>
              </svg>
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
      <div className="board-hint" aria-live="polite">
        {info}
      </div>
    </div>
  );
}
