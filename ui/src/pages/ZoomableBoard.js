import React, { useEffect, useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import clsx from "clsx";

import useWindowSize from "../utils/useWindowSize";
import { SQRT3 } from "../utils/coordinates";
import Tile from "./Tile";
import Node from "./Node";
import Edge from "./Edge";
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

export default function ZoomableBoard({ state }) {
  const { width, height } = useWindowSize();
  const [show, setShow] = useState(false);

  // TODO: Keep in sync with CSS
  const containerHeight = height - 144 - 120;

  const center = [width / 2, containerHeight / 2];
  const size = computeDefaultSize(width, containerHeight);

  const tiles = state.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={coordinate}
      center={center}
      coordinate={coordinate}
      tile={tile}
      size={size}
    />
  ));
  const nodes = Object.values(
    state.nodes
  ).map(({ color, building, direction, tile_coordinate, id }) => (
    <Node
      id={id}
      key={id}
      center={center}
      size={size}
      coordinate={tile_coordinate}
      direction={direction}
      building={building}
      color={color}
    />
  ));
  const edges = Object.values(
    state.edges
  ).map(({ color, direction, tile_coordinate, id }) => (
    <Edge
      id={id}
      key={id}
      center={center}
      size={size}
      coordinate={tile_coordinate}
      direction={direction}
      color={color}
    />
  ));

  useEffect(() => {
    setTimeout(() => {
      setShow(true);
    }, 300);
  }, []);

  return (
    <TransformWrapper
      options={{
        limitToBounds: true,
      }}
    >
      {({
        zoomIn,
        zoomOut,
        resetTransform,
        positionX,
        positionY,
        scale,
        previousScale,
      }) => (
        <React.Fragment>
          <div className="board-container">
            {
              <TransformComponent>
                <div className={clsx("board", { show })}>
                  {tiles}
                  {edges}
                  {nodes}
                  <Robber
                    center={center}
                    size={size}
                    coordinate={state.robber_coordinate}
                  />
                </div>
              </TransformComponent>
            }
          </div>
        </React.Fragment>
      )}
    </TransformWrapper>
  );
}
