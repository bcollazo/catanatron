import React, { useRef, useEffect, useState } from "react";

import useWindowSize from "../utils/useWindowSize";
import Tile from "./Tile";
import { SQRT3 } from "../utils/coordinates";
import Edge from "./Edge";
import Node from "./Node";
import Robber from "./Robber";

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
    // boardHeight = numLevels * (3h/4) + (1h/4)
    // 4*boardHeight = numLevels * 3h + 1h
    // 4*boardHeight = (3*numLevels + 1)*h, so:
    h = (4 * boardHeight) / (3 * numLevels + 1);
    size = h / 2;
    w = SQRT3 * size;
  }

  const tiles = state.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={coordinate}
      centerX={centerX}
      centerY={centerY}
      w={w}
      h={h}
      coordinate={coordinate}
      tile={tile}
      size={size}
    />
  ));
  const edges = Object.values(
    state.edges
  ).map(({ building, direction, tile_coordinate, id }) => (
    <Edge
      id={id}
      key={id}
      centerX={centerX}
      centerY={centerY}
      w={w}
      h={h}
      size={size}
      coordinate={tile_coordinate}
      direction={direction}
      building={building}
    />
  ));
  const nodes = Object.values(
    state.nodes
  ).map(({ building, direction, tile_coordinate, id }) => (
    <Node
      id={id}
      key={id}
      centerX={centerX}
      centerY={centerY}
      w={w}
      h={h}
      size={size}
      coordinate={tile_coordinate}
      direction={direction}
      building={building}
    />
  ));

  return (
    <div className="board-container flex-grow flex">
      <div ref={ref} className="board relative w-full h-full">
        {tiles}
        {edges}
        {nodes}
        <Robber
          centerX={centerX}
          centerY={centerY}
          w={w}
          h={h}
          size={size}
          coordinate={state.robber_coordinate}
        />
      </div>
    </div>
  );
}
