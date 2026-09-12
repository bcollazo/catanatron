import { ArrowLeftRight } from "lucide-react";
import {
  type GameState,
  type Tile,
  portGeometry,
  resourceColors,
  title,
} from "./model";
import { ResourceIcon } from "./Resources";

export function Port({
  game,
  port,
  onInspect,
}: {
  game: GameState;
  port: Tile;
  onInspect: (description: string) => void;
}) {
  const geometry = portGeometry(game, port);
  if (!geometry) return null;
  const {
    badge: [x, y],
    ends,
    nodes,
  } = geometry;
  const resource = port.tile.resource;
  const color = resource ? resourceColors[resource] : "var(--accent)";
  const description = `${resource ? `2:1 ${title(resource)}` : "3:1 Any resource"} port · connects nodes ${nodes.join(" and ")}`;
  return (
    <g className="port-connection" data-port-nodes={nodes.join(",")}>
      <g
        pointerEvents="none"
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeOpacity=".75"
      >
        {ends.map(([px, py], i) => (
          <g key={nodes[i]}>
            <path d={`M${px} ${py} L${x} ${y}`} className="port-link" />
            <circle
              cx={px}
              cy={py}
              r="3.5"
              fill="var(--panel)"
              data-port-node={nodes[i]}
            />
          </g>
        ))}
      </g>
      <g
        className="port-badge"
        transform={`translate(${x} ${y})`}
        role="button"
        tabIndex={0}
        aria-label={description}
        onClick={() => onInspect(description)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onInspect(description);
          }
        }}
      >
        <circle r="23" fill="transparent" />
        <circle r="16" fill="var(--panel)" stroke={color} strokeOpacity=".6" />
        <g transform="translate(-7 -12)" color={color}>
          {resource ? (
            <ResourceIcon resource={resource} size={14} />
          ) : (
            <ArrowLeftRight size={14} />
          )}
        </g>
        <text y="11" className="port-label" fill={color}>
          {resource ? "2:1" : "3:1"}
        </text>
        <title>{description}</title>
      </g>
    </g>
  );
}
