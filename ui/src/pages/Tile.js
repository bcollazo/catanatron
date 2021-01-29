import React from "react";
import cn from "classnames";

import brickTile from "../assets/tiles/tile_brick.png";
import desertTile from "../assets/tiles/tile_desert.svg";
import grainTile from "../assets/tiles/tile_grain.svg";
import lumberTile from "../assets/tiles/tile_lumber.svg";
import oreTile from "../assets/tiles/tile_ore.svg";
import woolTile from "../assets/tiles/tile_wool.svg";
import { tilePixelVector } from "../utils/coordinates";

const bgColorResource = (resource) => {
  return {
    SHEEP: "bg-green-200",
    WOOD: "bg-green-800",
    BRICK: "bg-red-400",
    ORE: "bg-gray-600",
    WHEAT: "bg-yellow-500",
  }[resource];
};

export function Circle({ className, children, style }) {
  return (
    <div
      className={cn(
        "rounded-full h-8 w-8 bg-white flex justify-center items-center border-2 border-black",
        className
      )}
      style={style}
    >
      {children}
    </div>
  );
}

export default function Tile({
  centerX,
  centerY,
  w,
  h,
  coordinate,
  tile,
  size,
}) {
  const [x, y] = tilePixelVector(coordinate, size, centerX, centerY);

  let contents;
  let resourceTile;
  if (tile.type === "RESOURCE_TILE") {
    contents = <Circle>{tile.number}</Circle>;
    resourceTile = {
      BRICK: brickTile,
      SHEEP: woolTile,
      ORE: oreTile,
      WOOD: lumberTile,
      WHEAT: grainTile,
    }[tile.resource];
  } else if (tile.type === "DESERT") {
    resourceTile = desertTile;
  } else if (tile.type === "PORT") {
    if (tile.resource === null) {
      contents = <Circle>3:1</Circle>;
    } else {
      const bg = bgColorResource(tile.resource);
      contents = <Circle className={bg}>2:1</Circle>;
    }
  }

  return (
    <div
      key={coordinate}
      className={cn("tile absolute flex justify-center items-center")}
      style={{
        left: x - w / 2,
        top: y - h / 2,
        width: w,
        height: h,
        backgroundImage: `url('${resourceTile}')`,
        backgroundSize: "contain",
      }}
    >
      {contents}
    </div>
  );
}
