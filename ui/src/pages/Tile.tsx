import cn from "classnames";
import Paper from "@mui/material/Paper";

import "./Tile.scss";
import brickTile from "../assets/tile_brick.svg";
import desertTile from "../assets/tile_desert.svg";
import grainTile from "../assets/tile_wheat.svg";
import lumberTile from "../assets/tile_wood.svg";
import oreTile from "../assets/tile_ore.svg";
import woolTile from "../assets/tile_sheep.svg";
import maritimeTile from "../assets/tile_maritime.svg";
import { SQRT3, tilePixelVector } from "../utils/coordinates";
import {
  type Direction,
  type ResourceCard,
  type Tile,
} from "../utils/api.types";

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
  return (
    <Paper
      elevation={3}
      className={cn("number-token", className, { flashing: flashing })}
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

const RESOURCES: { [K in ResourceCard]: string } = {
  BRICK: brickTile,
  SHEEP: woolTile,
  ORE: oreTile,
  WOOD: lumberTile,
  WHEAT: grainTile,
} as const;

const calculatePortPosition = (
  direction: Direction,
  size: number
): { x: number; y: number } => {
  let x = 0;
  let y = 0;
  if (direction.includes("SOUTH")) {
    y += size / 3;
  } else if (direction.includes("NORTH")) {
    y -= size / 3;
  }
  if (direction.includes("WEST")) {
    x -= size / 4;
    if (direction === "WEST") {
      x = -size / 3;
    }
  } else if (direction.includes("EAST")) {
    x += size / 4;
    if (direction === "EAST") {
      x = size / 3;
    }
  }
  return { x, y };
};

const Port = ({
  resource,
  style,
}: {
  resource: ResourceCard;
  style: Partial<React.CSSProperties>;
}) => {
  let ratio;
  let tile;
  if (resource in RESOURCES) {
    ratio = "2:1";
    tile = RESOURCES[resource];
  } else {
    ratio = "3:1";
    tile = maritimeTile;
  }

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
  const w = SQRT3 * size;
  const h = 2 * size;
  const [centerX, centerY] = center;
  const [x, y] = tilePixelVector(coordinate, size, centerX, centerY);

  let contents;
  let resourceTile;
  if (tile.type === "RESOURCE_TILE") {
    contents = <NumberToken number={tile.number} flashing={flashing} />;
    resourceTile = RESOURCES[tile.resource];
  } else if (tile.type === "DESERT") {
    resourceTile = desertTile;
  } else if (tile.type === "PORT") {
    const { x, y } = calculatePortPosition(tile.direction, size);
    contents = (<Port resource={tile.resource} style={{ left: x, top: y }}  />)
    }

  return (
    <div
      key={coordinate}
      className="tile"
      style={{
        left: x - w / 2,
        top: y - h / 2,
        width: w,
        height: h,
        backgroundImage: `url("${resourceTile}")`,
        backgroundSize: "contain",
      }}
      onClick={onClick}
    >
      {contents}
    </div>
  );
}
