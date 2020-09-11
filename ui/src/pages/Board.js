import React, { useRef, useEffect, useState } from "react";
import cn from "classnames";

import useWindowSize from "../utils/useWindowSize";
import Tile from "./Tile";
import {
  tilePixelVector,
  getNodeDelta,
  SQRT3,
  getEdgeDeltaAndTransform,
} from "../utils/coordinates";

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
    Tile(centerX, centerY, w, h, coordinate, tile, size)
  );
  const roads = [];
  Object.values(state.edges).forEach(
    ({ building, direction, tile_coordinate }) => {
      if (building !== null) {
        roads.push(
          renderRoad(
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
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY] = getNodeDelta(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;

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

const renderRoad = (
  centerX,
  centerY,
  w,
  h,
  size,
  coordinate,
  direction,
  building
) => {
  const color = `bg-white bg-${building.color.toLowerCase()}-700`;
  const [tileX, tileY] = tilePixelVector(coordinate, size, centerX, centerY);
  const [deltaX, deltaY, transform] = getEdgeDeltaAndTransform(direction, w, h);
  const x = tileX + deltaX;
  const y = tileY + deltaY;
  return (
    <div
      className={cn(
        "road absolute border-2 border-black h-3 w-10",
        color,
        coordinate,
        direction
      )}
      style={{
        left: x,
        top: y,
        transform: transform,
      }}
    ></div>
  );
};
