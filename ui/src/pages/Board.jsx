import React from "react";
import classnames from "classnames";

import { SQRT3 } from "../utils/coordinates";
import Tile from "./Tile";
import Node from "./Node";
import Edge from "./Edge";
import Robber from "./Robber";

import "./Board.scss";

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

export default function Board({
  width,
  height,
  buildOnNodeClick,
  buildOnEdgeClick,
  handleTileClick,
  nodeActions,
  edgeActions,
  replayMode,
  gameState,
  isMobile,
  show,
  isMovingRobber,
}) {
  // TODO: Keep in sync with CSS
  const containerHeight = height - 144 - 38 - 40;
  const containerWidth = isMobile ? width - 280 : width;
  const center = [containerWidth / 2, containerHeight / 2];
  const size = computeDefaultSize(containerWidth, containerHeight);
  if (!size) {
    return null;
  }

  const tiles = gameState.tiles.map(({ coordinate, tile }) => (
    <Tile
      key={coordinate}
      center={center}
      coordinate={coordinate}
      tile={tile}
      size={size}
      flashing={isMovingRobber}
      onClick={() => handleTileClick(coordinate)}
    />
  ));
  const nodes = Object.values(gameState.nodes).map(
    ({ color, building, direction, tile_coordinate, id }) => (
      <Node
        key={id}
        id={id}
        center={center}
        size={size}
        coordinate={tile_coordinate}
        direction={direction}
        building={building}
        color={color}
        flashing={!replayMode && id in nodeActions}
        onClick={buildOnNodeClick(id, nodeActions[id])}
      />
    )
  );
  const edges = Object.values(gameState.edges).map(
    ({ color, direction, tile_coordinate, id }) => (
      <Edge
        id={id}
        key={id}
        center={center}
        size={size}
        coordinate={tile_coordinate}
        direction={direction}
        color={color}
        flashing={id in edgeActions}
        onClick={buildOnEdgeClick(id, edgeActions[id])}
      />
    )
  );
  return (
    <div className={classnames("board", { show })}>
      {tiles}
      {edges}
      {nodes}
      <Robber
        center={center}
        size={size}
        coordinate={gameState.robber_coordinate}
      />
    </div>
  );
}
