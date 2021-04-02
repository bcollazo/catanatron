import React, { useRef, useEffect, useState, useCallback } from "react";

import useWindowSize from "../utils/useWindowSize";
import Tile from "./Tile";
import { SQRT3 } from "../utils/coordinates";
import Edge from "./Edge";
import Node from "./Node";
import Robber from "./Robber";

/**
 * This uses the formulas: W = SQRT3 * size and H = 2 * size.
 * Math comes from https://www.redblobgames.com/grids/hexagons/.
 */
function computeDefaultSize(divWidth, divHeight) {
  const numLevels = 6; // 3 rings + 1/2 a tile for the outer water ring
  // divHeight = numLevels * (3h/4) + (h/4), implies:
  const maxSizeThatRespectsHeight = (4 * divHeight) / (3 * numLevels + 1) / 2; 
  const correspondingWidth = SQRT3 * maxSizeThatRespectsHeight;
  let size;
  if (numLevels * correspondingWidth < divWidth) {
    // thus complete board would fit if we pick size based on height (height is limiting factor)
    size = maxSizeThatRespectsHeight;
  } else {
    // we'll have to decide size based on width.
    const maxSizeThatRespectsWidth = divWidth / numLevels / SQRT3;
    size = maxSizeThatRespectsWidth;
  }
  return size;
}

export default function Board({ state }) {
  const { width, height } = useWindowSize();
  const ref = useRef(null);
  const [containerWidth, setContainerWidth] = useState(null);
  const [containerHeight, setContainerHeight] = useState(null);
  const [size, setSize] = useState(null);

  // Set Board Width when we get our computed flex space available.
  useEffect(() => {
    if (ref.current !== null) {
      const style = window.getComputedStyle(ref.current, null);
      const divWidth = parseInt(style.getPropertyValue("width"));
      const divHeight = parseInt(style.getPropertyValue("height"));
      setContainerWidth(divWidth);
      setContainerHeight(divHeight);
      setSize(computeDefaultSize(divWidth, divHeight))
    }
  }, [ref, width, height]);

  useEffect(() => {
    const handleWheel = (event) => {
      console.log(size, event.deltaY);
      // TODO: CAP
      const newSize = size + event.deltaY * -0.1; // deltaY < 0 means bigger
      setSize(newSize);
    }
    window.addEventListener('wheel', handleWheel, { passive: true })
    return () => {
      window.removeEventListener('wheel', handleWheel)
    }
  }, [size]);

  if (containerWidth === null || containerHeight === null) {
    return <div className="board-container flex-grow flex">
      <div ref={ref} className="board relative w-full h-full">
      </div>
    </div>;
  }

  const centerX = containerWidth / 2;
  const centerY = containerHeight / 2;
  const w = SQRT3 * size;
  const h = 2 * size;
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
  // const edges = Object.values(
  //   state.edges
  // ).map(({ color, direction, tile_coordinate, id }) => (
  //   <Edge
  //     id={id}
  //     key={id}
  //     centerX={centerX}
  //     centerY={centerY}
  //     w={w}
  //     h={h}
  //     scalingFactor={scalingFactor}
  //     size={size}
  //     coordinate={tile_coordinate}
  //     direction={direction}
  //     color={color}
  //   />
  // ));
  // const nodes = Object.values(
  //   state.nodes
  // ).map(({ color, building, direction, tile_coordinate, id }) => (
  //   <Node
  //     id={id}
  //     key={id}
  //     centerX={centerX}
  //     centerY={centerY}
  //     w={w}
  //     h={h}
  //     size={size}
  //     coordinate={tile_coordinate}
  //     direction={direction}
  //     building={building}
  //     color={color}
  //   />
  // ));

  return (
    <div className="board-container flex-grow flex">
      <div ref={ref} className="board relative w-full h-full" onDrag={onDrag}>
        {tiles}
        {/* {edges}
        {nodes} */}
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
