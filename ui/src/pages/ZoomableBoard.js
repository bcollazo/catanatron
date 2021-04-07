import React, { useRef, useEffect, useState, useCallback } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

import useWindowSize from "../utils/useWindowSize";
import { SQRT3 } from "../utils/coordinates";
import Tile from "./Tile";
import ActionsToolbar from "./ActionsToolbar";

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

  const center = [width / 2, height / 2];
  const size = computeDefaultSize(width, height);

  const tiles = state.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={coordinate}
      center={center}
      coordinate={coordinate}
      tile={tile}
      size={size}
    />
  ));

  return (
    <TransformWrapper
      options={{
        limitToBounds: false,
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
                <div className="example-text">
                  {/* <h1>Block</h1> */}
                  {tiles}
                  {/* <Robber
                        center={[positionX + center[0], positionY + center[1]]}
                        w={w}
                        h={h}
                        size={size}
                        coordinate={state.robber_coordinate}
                      /> */}
                </div>
              </TransformComponent>
            }
          </div>
          <ActionsToolbar zoomIn={zoomIn} zoomOut={zoomOut} />
        </React.Fragment>
      )}
    </TransformWrapper>
  );
}
