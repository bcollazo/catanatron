import {
  Trees,
  BrickWall,
  Wheat,
  Stone,
  Hexagon,
  Shield,
  Crown,
  Layers,
  Route,
  Sparkles,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  RESOURCES,
  resourceColors,
  stat,
  center,
  nodePoint,
  same,
  title,
  type GameState,
  type Color,
  type Resource,
} from "./model";
function SheepIcon({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 17v4m9-4v4M4 9a3 3 0 0 1 4-4 3 3 0 0 1 5 0 3 3 0 0 1 4 4v4a4 4 0 0 1-4 4H6a4 4 0 0 1-2-8Z" />
      <path d="m17 8 3-1 2 3-1 6h-4l-1-5M19 11h.01" />
    </svg>
  );
}
export const resourceIcons = {
  WOOD: Trees,
  BRICK: BrickWall,
  SHEEP: SheepIcon,
  WHEAT: Wheat,
  ORE: Stone,
};
export const devCards = {
  KNIGHT: Shield,
  MONOPOLY: Layers,
  YEAR_OF_PLENTY: Sparkles,
  ROAD_BUILDING: Route,
  VICTORY_POINT: Crown,
};
type DevelopmentCard = keyof typeof devCards;
type HandCardType = Resource | DevelopmentCard;
const developmentTypes = Object.keys(devCards) as DevelopmentCard[];
const isResource = (card: HandCardType): card is Resource =>
  RESOURCES.includes(card as Resource);
function HandCardFace({ card }: { card: HandCardType }) {
  if (isResource(card)) return <CardFace resource={card} />;
  const Icon = devCards[card];
  return (
    <>
      <span className="card-corner">
        <Icon size={12} />
      </span>
      <span className="card-emblem">
        <Icon size={28} />
      </span>
      <span className="card-name">{title(card)}</span>
    </>
  );
}
export function ResourceIcon({
  resource,
  size = 22,
}: {
  resource?: Resource | null;
  size?: number;
}) {
  const Icon = resource ? resourceIcons[resource] : Hexagon;
  return <Icon size={size} aria-hidden="true" />;
}
export function CardFace({ resource }: { resource: Resource }) {
  return (
    <>
      <span className="card-corner">
        <ResourceIcon resource={resource} size={12} />
      </span>
      <span className="card-emblem">
        <ResourceIcon resource={resource} size={28} />
      </span>
      <span className="card-name">{title(resource)}</span>
    </>
  );
}
export function productionSource(
  game: GameState,
  color: Color,
  resource: Resource,
) {
  const last = game.action_records.at(-1);
  if (last?.[0][1] !== "ROLL" || !Array.isArray(last[1])) return undefined;
  const total = last[1][0] + last[1][1];
  return game.tiles.find(
    (t) =>
      t.tile.resource === resource &&
      t.tile.number === total &&
      !same(t.coordinate, game.robber_coordinate) &&
      Object.values(game.nodes).some((n) => {
        if (n.color !== color || !n.building) return false;
        const a = nodePoint(n),
          b = center(t.coordinate);
        return Math.abs(Math.hypot(a[0] - b[0], a[1] - b[1]) - 58) < 0.001;
      }),
  )?.coordinate;
}
function HandCard({
  resource,
  color,
  game,
  arriving,
  compact,
  order,
}: {
  resource: HandCardType;
  color: Color;
  game: GameState;
  arriving: boolean;
  compact: boolean;
  order: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const [flight, setFlight] = useState<{
    fromX: number;
    fromY: number;
    toX: number;
    toY: number;
    width: number;
    height: number;
  } | null>(null);
  useLayoutEffect(() => {
    if (!arriving || reduced || !ref.current) return;
    const target = ref.current.getBoundingClientRect();
    if (!target.width || !target.height) return;
    if (
      compact &&
      document
        .querySelector(`[data-primary-hand="${color}"]`)
        ?.getBoundingClientRect().width
    )
      return;
    const tile = isResource(resource)
      ? productionSource(game, color, resource)
      : undefined;
    const source = tile
      ? document.querySelector(`[data-tile-coordinate="${tile.join(",")}"]`)
      : document.querySelector(".board-stage");
    const origin = source?.getBoundingClientRect();
    if (!origin?.width || !origin.height) return;
    const viewport = ref.current.parentElement!.getBoundingClientRect();
    setFlight({
      fromX: origin.left + origin.width / 2 - target.width / 2,
      fromY: origin.top + origin.height / 2 - target.height / 2,
      toX: Math.max(
        viewport.left,
        Math.min(viewport.right - target.width, target.left),
      ),
      toY: target.top,
      width: target.width,
      height: target.height,
    });
    // A flight belongs to this physical card's arrival, not subsequent game ticks.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const style = {
    "--resource": isResource(resource) ? resourceColors[resource] : "#b9a0ff",
  } as React.CSSProperties;
  return (
    <>
      <motion.div
        ref={ref}
        className={`hand-card ${isResource(resource) ? "" : "development-card"}`}
        data-resource-card={isResource(resource) ? resource : undefined}
        data-development-card={isResource(resource) ? undefined : resource}
        role="listitem"
        aria-label={title(resource)}
        title={title(resource)}
        style={style}
        initial={false}
        animate={{ opacity: flight ? 0 : 1 }}
        exit={{ opacity: 0, y: -28, rotate: 8, scale: 0.85 }}
        transition={{ duration: 0.22 }}
      >
        <HandCardFace card={resource} />
      </motion.div>
      {flight &&
        createPortal(
          <motion.div
            className={`hand-card flying-card ${compact ? "flying-compact" : ""} ${isResource(resource) ? "" : "development-card"}`}
            aria-hidden="true"
            style={{
              ...style,
              position: "fixed",
              left: 0,
              top: 0,
              width: flight.width,
              height: flight.height,
            }}
            initial={{
              x: flight.fromX,
              y: flight.fromY,
              rotate: -14,
              scale: 0.8,
              opacity: 0,
            }}
            animate={{
              x: flight.toX,
              y: flight.toY,
              rotate: 0,
              scale: 1,
              opacity: [0, 1, 1],
            }}
            transition={{
              duration: 0.65,
              delay: Math.min(order, 5) * 0.055,
              ease: [0.22, 0.68, 0.3, 1],
            }}
            onAnimationComplete={() => setFlight(null)}
          >
            <HandCardFace card={resource} />
          </motion.div>,
          document.body,
        )}
    </>
  );
}
export function ResourceHand({
  game,
  color,
  compact = false,
}: {
  game: GameState;
  color: Color;
  compact?: boolean;
}) {
  const cardTypes: HandCardType[] = compact
    ? [...RESOURCES]
    : [...RESOURCES, ...developmentTypes];
  const counts = Object.fromEntries(
    cardTypes.map((r) => [r, stat(game, color, `${r}_IN_HAND`)]),
  ) as Record<HandCardType, number>;
  const previous = useRef({ counts, color, index: game.state_index });
  const isNext =
    previous.current.color === color &&
    game.state_index === previous.current.index + 1;
  const cards = cardTypes.flatMap((resource) =>
    Array.from({ length: counts[resource] }, (_, i) => ({
      resource,
      id: `${color}-${resource}-${i}`,
      arriving: isNext && i >= previous.current.counts[resource],
    })),
  );
  useLayoutEffect(() => {
    previous.current = { counts, color, index: game.state_index };
  });
  return (
    <div
      className={`resource-hand ${compact ? "compact" : ""}`}
      data-primary-hand={compact ? undefined : color}
      aria-label={`${title(color)} hand · ${cards.length} cards`}
      role="list"
      tabIndex={0}
    >
      <AnimatePresence initial={false}>
        {cards.map((card, i) => (
          <HandCard
            key={card.id}
            {...card}
            order={i}
            game={game}
            color={color}
            compact={compact}
          />
        ))}
      </AnimatePresence>
      {!cards.length && (
        <span className="empty-hand">
          {compact ? "No resource cards" : "No cards in hand"}
        </span>
      )}
    </div>
  );
}
export function Legend() {
  return (
    <div className="legend">
      {RESOURCES.map((r) => (
        <span key={r} style={{ color: resourceColors[r] }}>
          <ResourceIcon resource={r} size={16} />
          {title(r)}
        </span>
      ))}
    </div>
  );
}
