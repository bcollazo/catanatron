import cn from "classnames";
import Paper from "@mui/material/Paper";

import "./Tile.scss";
import { SQRT3, tilePixelVector } from "../utils/coordinates";
import {
  type Direction,
  type ResourceCard,
  type Tile,
} from "../utils/api.types";
import { useSkin, type SkinAssets, type SkinName } from "../SkinContext";

type ResourceType = "WOOD" | "BRICK" | "SHEEP" | "WHEAT" | "ORE";

const RESOURCE_SYMBOL_KEYS: Record<ResourceType, "WOOD_SYMBOL" | "BRICK_SYMBOL" | "SHEEP_SYMBOL" | "WHEAT_SYMBOL" | "ORE_SYMBOL"> = {
  WOOD: "WOOD_SYMBOL",
  BRICK: "BRICK_SYMBOL",
  SHEEP: "SHEEP_SYMBOL",
  WHEAT: "WHEAT_SYMBOL",
  ORE: "ORE_SYMBOL",
};

type NumberTokenProps = {
  number: number;
  className?: string;
  style?: Partial<React.CSSProperties>;
  flashing?: boolean;
};

export function NumberToken({
  number,
  className,
  style,
  flashing,
}: NumberTokenProps) {
  const isHot = number === 6 || number === 8;
  return (
    <Paper
      elevation={3}
      className={cn("number-token", className, { flashing, hot: isHot })}
      style={style}
    >
      <div>{number}</div>
      <div className="pips">{numberToPips(number)}</div>
    </Paper>
  );
}

const numberToPips = (number: number) => {
  switch (number) {
    case 2:
    case 12:
      return "•";
    case 3:
    case 11:
      return "••";
    case 4:
    case 10:
      return "•••";
    case 5:
    case 9:
      return "••••";
    case 6:
    case 8:
      return "•••••";
    default:
      return "";
  }
};

const getResourceTiles = (assets: SkinAssets): { [K in ResourceCard]: string } => ({
  BRICK: assets.BRICK,
  SHEEP: assets.SHEEP,
  ORE: assets.ORE,
  WOOD: assets.WOOD,
  WHEAT: assets.WHEAT,
});

const calculatePortPosition = (
  direction: Direction,
  size: number,
  skin: SkinName = "catanatron"
): { x: number; y: number } => {
  // Classic ports are pushed further toward the vertices they serve
  const factor = skin === "classic" ? 1.25 : 1;
  let x = 0;
  let y = 0;
  if (direction.includes("SOUTH")) {
    y += (size / 3) * factor;
  } else if (direction.includes("NORTH")) {
    y -= (size / 3) * factor;
  }
  if (direction.includes("WEST")) {
    x -= (size / 4) * factor;
    if (direction === "WEST") {
      x = (-size / 3) * factor;
    }
  } else if (direction.includes("EAST")) {
    x += (size / 4) * factor;
    if (direction === "EAST") {
      x = (size / 3) * factor;
    }
  }
  return { x, y };
};

const Port = ({
  resource,
  style,
  assets,
  skin,
}: {
  resource: ResourceCard;
  style: Partial<React.CSSProperties>;
  assets: SkinAssets;
  skin: SkinName;
}) => {
  const RESOURCES = getResourceTiles(assets);
  const isSpecific = resource in RESOURCES;

  if (skin === "classic" && assets.TRADING_PORT) {
    const symbolKey = isSpecific
      ? RESOURCE_SYMBOL_KEYS[resource as ResourceType]
      : undefined;
    const symbolSrc = symbolKey ? assets[symbolKey] : undefined;

    return (
      <div className="port classic-port" style={style}>
        <img
          src={assets.TRADING_PORT}
          alt="port"
          className="classic-port-boat"
        />
        <div className="classic-port-sail">
          {symbolSrc ? (
            <img src={symbolSrc} alt={resource} className="classic-port-resource" />
          ) : (
            <span className="classic-port-ratio">3:1</span>
          )}
        </div>
      </div>
    );
  }

  const tile = isSpecific ? RESOURCES[resource] : assets.MARITIME;
  const ratio = isSpecific ? "2:1" : "3:1";

  return (
    <div
      className="port"
      style={{
        ...style,
        backgroundImage: `url("${tile}")`,
        height: 60,
        backgroundSize: "contain",
        width: 52,
        backgroundRepeat: "no-repeat",
      }}
    >
      {ratio}
    </div>
  );
};

type TileProps = {
  center: any;
  coordinate: any;
  tile: Tile;
  size: any;
  onClick: React.MouseEventHandler<HTMLDivElement>;
  flashing: boolean;
};

export default function Tile({
  center,
  coordinate,
  tile,
  size,
  onClick,
  flashing,
}: TileProps) {
  const { skin, assets } = useSkin();
  const RESOURCES = getResourceTiles(assets);
  const w = SQRT3 * size;
  const h = 2 * size;
  const [centerX, centerY] = center;
  const [x, y] = tilePixelVector(coordinate, size, centerX, centerY);

  let contents;
  let resourceTile;
  let tileType = "";
  if (tile.type === "RESOURCE_TILE") {
    tileType = "tile-resource";
    contents = <NumberToken number={tile.number} flashing={flashing} />;
    resourceTile = RESOURCES[tile.resource];
  } else if (tile.type === "DESERT") {
    tileType = "tile-desert";
    resourceTile = assets.DESERT;
  } else if (tile.type === "PORT") {
    tileType = "tile-port";
    const { x, y } = calculatePortPosition(tile.direction, size, skin);
    contents = (<Port resource={tile.resource} style={{ left: x, top: y }} assets={assets} skin={skin} />)
    }

  return (
    <div
      key={coordinate}
      className={cn("tile", `skin-${skin}`, tileType)}
      style={{
        left: x - w / 2,
        top: y - h / 2,
        width: w,
        height: h,
        ...(resourceTile
          ? { backgroundImage: `url("${resourceTile}")`, backgroundSize: "contain" }
          : {}),
      }}
      onClick={onClick}
    >
      {contents}
    </div>
  );
}
