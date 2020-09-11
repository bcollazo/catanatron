import React, { useRef, useEffect, useState } from "react";
import cn from "classnames";

import useWindowSize from "../utils/useWindowSize";
import brickTile from "../assets/tiles/tile_brick.svg";
import desertTile from "../assets/tiles/tile_desert.svg";
import grainTile from "../assets/tiles/tile_grain.svg";
import lumberTile from "../assets/tiles/tile_lumber.svg";
import oreTile from "../assets/tiles/tile_ore.svg";
import woolTile from "../assets/tiles/tile_wool.svg";

// https://www.redblobgames.com/grids/hexagons/
const SQRT3 = 1.73205080757;

const bgColorResource = (resource) => {
  return {
    SHEEP: "bg-green-200",
    WOOD: "bg-green-800",
    BRICK: "bg-red-400",
    ORE: "bg-gray-600",
    WHEAT: "bg-yellow-500",
  }[resource];
};

export default function Board({ state }) {
  const ref = useRef(null);
  const [boardWidth, setBoardWidth] = useState(0);
  const [boardHeight, setBoardHeight] = useState(0);
  const { width, height } = useWindowSize();

  useEffect(() => {
    if (ref.current !== null) {
      const style = window.getComputedStyle(ref.current, null);
      setBoardWidth(parseInt(style.getPropertyValue("width")));
      setBoardHeight(parseInt(style.getPropertyValue("height")));
    }
  }, [ref, width, height]);

  // signed integer to create zoom effect. 1 means it should be contained.
  const boardScalingFactor = 1;
  const centerX = (boardWidth * boardScalingFactor) / 2;
  const centerY = (boardHeight * boardScalingFactor) / 2;

  const numLevels = 7; // 2 outermost of water, 3 inner layers
  let w, size, h;
  // if assuming width, would break height
  if ((2 * boardWidth) / SQRT3 < boardHeight) {
    w = boardWidth / numLevels;
    size = w / SQRT3;
    h = 2 * size;
  } else {
    h = (4 * (boardHeight / numLevels)) / 3;
    size = h / 2;
    w = SQRT3 * size;
  }

  const tiles = state.tiles.map(({ coordinate, tile }) =>
    renderTile(centerX, centerY, w, h, coordinate, tile, size)
  );
  const roads = [];
  Object.entries(state.edges).forEach(
    ([index, { building, direction, tile_coordinate }]) => {
      if (building !== null) {
        roads.push(renderRoad());
      }
    }
  );
  const nodeBuildings = [];
  Object.values(state.nodes).forEach(
    ({ building, direction, tile_coordinate }) => {
      if (building !== null) {
        nodeBuildings.push(
          renderNodeBuilding(
            centerX,
            centerY,
            w,
            h,
            size,
            tile_coordinate,
            direction,
            building
          )
        );
      }
    }
  );

  return (
    <div className="board-container w-full flex-grow m-10">
      <div ref={ref} className="board relative w-full h-full">
        {tiles}
        {roads}
        {nodeBuildings}
      </div>
    </div>
  );
}

const getDelta = (direction, w, h) => {
  switch (direction) {
    case "NORTH":
      return [0, -h / 2];
    case "NORTHEAST":
      return [w / 2, -h / 4];
    case "SOUTHEAST":
      return [w / 2, h / 4];
    case "SOUTH":
      return [0, h / 2];
    case "SOUTHWEST":
      return [-w / 2, h / 4];
    case "NORTHWEST":
      return [-w / 2, -h / 4];
    default:
      throw Error("Unkown direction " + direction);
  }
};

const renderNodeBuilding = (
  centerX,
  centerY,
  w,
  h,
  size,
  coordinate,
  direction,
  building
) => {
  const [tileX, tileY] = tilePixelVector(coordinate, size);
  const [deltaX, deltaY] = getDelta(direction, w, h);
  const x = tileX + deltaX + centerX;
  const y = tileY + deltaY + centerY;

  const color = `bg-white bg-${building.color.toLowerCase()}-700`;
  return (
    <div
      className={cn(
        "node-building absolute w-6 h-6 border-2 border-black",
        color
      )}
      style={{
        left: x,
        top: y,
        transform: `translateY(-0.75rem) translateX(-0.75rem)`,
      }}
    ></div>
  );
};

const renderRoad = (coordinate, size) => {
  return <div className="road absolute">Road</div>;
};

const renderTile = (centerX, centerY, w, h, coordinate, tile, size) => {
  const [x, y] = tilePixelVector(coordinate, size);

  let contents;
  let resourceTile;
  if (tile.type === "RESOURCE_TILE") {
    contents = (
      <div className="rounded-full h-8 w-8 bg-white flex justify-center items-center border-2 border-black">
        {tile.number}
      </div>
    );
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
      contents = (
        <div className="rounded-full h-8 w-8 bg-white flex justify-center items-center border-2 border-black">
          3:1
        </div>
      );
    } else {
      const bg = bgColorResource(tile.resource);
      contents = (
        <div
          className={cn(
            "rounded-full h-8 w-8 bg-white flex justify-center items-center border-2 border-black",
            bg
          )}
        >
          2:1
        </div>
      );
    }
  }

  const margin = 0; // until tiles svgs arrive
  return (
    <div
      key={coordinate}
      className={cn("tile absolute flex justify-center items-center h-16 w-16")}
      style={{
        left: x + centerX,
        top: y + centerY,
        width: w - margin,
        height: h - margin,
        margin: margin,
        transform: `translateY(-${h / 2}px) translateX(-${w / 2}px)`,
        backgroundImage: `url('${resourceTile}')`,
        backgroundSize: "contain",
      }}
    >
      {contents}
    </div>
  );
};

const tilePixelVector = (coordinate, size) => {
  const hex = cubeToAxial(coordinate);
  return [size * (SQRT3 * hex.q + (SQRT3 / 2) * hex.r), size * (3 / 2) * hex.r];
};

function cubeToAxial(cube) {
  return { q: cube[0], r: cube[2] };
}
