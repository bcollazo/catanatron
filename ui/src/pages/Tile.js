import React from "react";
import cn from "classnames";

import brickTile from "../assets/tile_brick.png";
import desertTile from "../assets/tile_desert.png";
import grainTile from "../assets/tile_grain.png";
import lumberTile from "../assets/tile_lumber.png";
import oreTile from "../assets/tile_ore.png";
import woolTile from "../assets/tile_wool.png";
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
        "rounded-md h-8 w-8 bg-white flex justify-center items-center border-2 border-black",
        className
      )}
      style={style}
    >
      {children}
    </div>
  );
}

export default function Tile({
  center,
  w,
  h,
  coordinate,
  tile,
  size,
}) {
  const [centerX, centerY] = center;
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
    console.log(tile);
    let x = 0;
    let y = 0;
    if (tile.direction.includes("SOUTH")) {
      y += 50;
    }
    if (tile.direction.includes("NORTH")) {
      y -= 50;
    }
    if (tile.direction.includes("WEST")) {
      x -= 20;
      if (tile.direction === "WEST") {
        x = -50;
      }
    }
    if (tile.direction.includes("EAST")) {
      x += 20;
      if (tile.direction === "EAST") {
        x = 50;
      }
    }
    if (tile.resource === null) {
      contents = (
        <Circle
          className={tile.direction}
          style={{
            position: "relative",
            left: x,
            top: y,
          }}
        >
          3:1
        </Circle>
      );
    } else {
      const bg = bgColorResource(tile.resource);
      contents = (
        <Circle
          style={{
            position: "relative",
            left: x,
            top: y,
          }}
          className={bg}
        >
          2:1
        </Circle>
      );
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
