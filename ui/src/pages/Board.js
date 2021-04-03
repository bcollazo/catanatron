import React, { useRef, useEffect, useState, useCallback } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

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
  const [draggingStart, setDraggingStart] = useState(null); // null if not dragging, the start coord else
  const [center, setCenter] = useState(null);

  const onMouseDown = (event) => {
    console.log(event);
  };

  // Set Board Width when we get our computed flex space available.
  useEffect(() => {
    if (ref.current !== null) {
      const style = window.getComputedStyle(ref.current, null);
      const divWidth = parseInt(style.getPropertyValue("width"));
      const divHeight = parseInt(style.getPropertyValue("height"));
      setContainerWidth(divWidth);
      setContainerHeight(divHeight);
      setSize(computeDefaultSize(divWidth, divHeight));
      setCenter([divWidth / 2, divHeight / 2]);
      console.log(divWidth, divHeight);
    }
  }, [ref, width, height]);

  useEffect(() => {
    const handleWheel = (event) => {
      console.log(size, event.deltaY);
      // TODO: CAP
      const newSize = size + event.deltaY * -0.1; // deltaY < 0 means bigger
      setSize(newSize);
    };
    window.addEventListener("wheel", handleWheel, { passive: true });
    return () => {
      window.removeEventListener("wheel", handleWheel);
    };
  }, [size]);

  return (
    <div className="board-container flex-grow flex">
      <div ref={ref} className="board relative w-full h-full"></div>
    </div>
  );
  if (containerWidth === null || containerHeight === null) {
  }

  console.log("center", center);
  const w = SQRT3 * size;
  const h = 2 * size;
  const tiles = state.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={coordinate}
      center={center}
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
    <div className="h-full">
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
            <div className="element">
              {
                <TransformComponent>
                  <div className="example-text">
                    <h1>Lorem ipsum</h1>
                    <p>
                      Lorem ipsum dolor sit amet, consectetur adipiscing elit,
                      sed do eiusmod tempor incididunt ut labore et dolore magna
                      aliqua. Ut enim ad minim veniam, quis nostrud exercitation
                      ullamco laboris nisi ut aliquip ex ea commodo consequat.
                      Duis aute irure dolor in reprehenderit in voluptate velit
                      esse cillum dolore eu fugiat nulla pariatur. Excepteur
                      sint occaecat cupidatat non proident, sunt in culpa qui
                      officia deserunt mollit anim id est laborum.
                    </p>
                    <h1>SVG</h1>
                    <Robber
                      center={[positionX + center[0], positionY + center[1]]}
                      w={w}
                      h={h}
                      size={size}
                      coordinate={state.robber_coordinate}
                    />
                  </div>
                </TransformComponent>
              }
            </div>
            <div className="tools">
              <div className="info">
                <h3>State</h3>
                <h5>
                  <span className="badge badge-secondary">
                    Position x : {positionX}px
                  </span>
                  <span className="badge badge-secondary">
                    Position y : {positionY}px
                  </span>
                  <span className="badge badge-secondary">Scale : {scale}</span>
                  <span className="badge badge-secondary">
                    Previous scale : {previousScale}
                  </span>
                </h5>
              </div>
              <button
                className="btn-gradient cyan small"
                onClick={zoomIn}
                data-testid="zoom-in-button"
              >
                Zoom In
              </button>
              <button
                className="btn-gradient blue small"
                onClick={zoomOut}
                data-testid="zoom-out-button"
              >
                Zoom Out
              </button>
              <button
                className="btn-gradient purple small"
                onClick={resetTransform}
                data-testid="reset-button"
              >
                Reset
              </button>
            </div>
          </React.Fragment>
        )}
      </TransformWrapper>
    </div>
  );
  // {
  //   /* <div className="board-container flex-grow flex">
  //         <div
  //           ref={ref}
  //           className="board relative w-full h-full"
  //           onMouseDown={onMouseDown}
  //         > */
  // }
  // {
  //   tiles;
  // }
  // {
  //   /* {edges}
  //       {nodes} */
  // }
  // <Robber
  //   center={center}
  //   w={w}
  //   h={h}
  //   size={size}
  //   coordinate={state.robber_coordinate}
  // />;
  // {
  //   /* </div>
  //       </div> */
  // }
}
