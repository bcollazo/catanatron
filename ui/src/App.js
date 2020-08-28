import React, { useEffect, useState, useRef } from "react";
import cn from "classnames";

import "./App.scss";
import useWindowSize from "./useWindowSize";
import { ControlPanel } from "./ControlPanel";

const URL = "http://localhost:5000/board";

function Tile({ resource, number }) {
  return (
    <div className={cn("hexagon", { [resource.toLowerCase()]: true })}>
      {number}
    </div>
  );
}

function DesertTile() {
  return <div className="hexagon desert"></div>;
}

function Water() {
  return <div className="hexagon water"></div>;
}

const write = (ctx, text, x, y, font = "30px Helvetica", color = "white") => {
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.fillText(text, x, y);
};

const drawTile = (ctx, x, y, radius, tile) => {
  ctx.fillStyle = color(tile);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fill();

  if (tile.number !== undefined) {
    write(ctx, tile.number, x - 1, y + 8);
  }
  if (tile.type === "PORT") {
    if (tile.resource === null) {
      write(ctx, "3:1", x - 1, y + 8);
    } else {
      ctx.fillStyle = colorResource(tile.resource);
      ctx.beginPath();
      ctx.arc(x, y, radius / 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
};

function drawNode(ctx, x, y, nodeId) {
  write(ctx, nodeId, x, y, "10px Helvetica", "black");
}

const color = (tile) => {
  if (tile.type === "WATER" || tile.type === "PORT") {
    return "#0000ff";
  } else if (tile.type === "DESERT") {
    return "orange";
  } else {
    return colorResource(tile.resource);
  }
};

const colorResource = (resource) => {
  return {
    SHEEP: "lightgreen",
    WOOD: "green",
    BRICK: "red",
    ORE: "gray",
    WHEAT: "yellow",
  }[resource];
};

// https://www.redblobgames.com/grids/hexagons/
const r = 60; // for drawing circle
const sqrt3 = 1.73205080757;
const size = 80;
const w = sqrt3 * size;
const h = 2 * size;
function cube_to_axial(cube) {
  return { q: cube[0], r: cube[2] };
}
const translateCoordinate = (coordinate, centerX, centerY) => {
  const hex = cube_to_axial(coordinate);
  return [
    centerX + size * (sqrt3 * hex.q + (sqrt3 / 2) * hex.r),
    centerY + size * (3 / 2) * hex.r,
  ];
};
const getDelta = (direction) => {
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
  }
};

const draw = (ctx, state, windowWidth, windowHeight) => {
  const centerX = windowWidth / 2;
  const centerY = windowHeight / 2;
  for (const { coordinate, tile } of state.tiles) {
    const [canvasX, canvasY] = translateCoordinate(
      coordinate,
      centerX,
      centerY
    );
    drawTile(ctx, canvasX, canvasY, r, tile);
  }

  for (const nodeId in state.nodes) {
    const node = state.nodes[nodeId];
    const [tileX, tileY] = translateCoordinate(
      node.tile_coordinate,
      centerX,
      centerY
    );
    const [deltaX, deltaY] = getDelta(node.direction);
    const x = tileX + deltaX;
    const y = tileY + deltaY;
    drawNode(ctx, x, y, nodeId);
  }
};

function resizeCanvas(canvas, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    const { devicePixelRatio: ratio = 1 } = window;
    const context = canvas.getContext("2d");
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.scale(ratio, ratio);
  }
}

function App() {
  const [state, setState] = useState(null);
  const { width, height } = useWindowSize();
  const canvasRef = useRef(null);

  useEffect(() => {
    (async () => {
      const response = await fetch(URL);
      const data = await response.json();
      setState(data);
    })();
  }, []);

  useEffect(() => {
    if (canvasRef.current === null) {
      return;
    }
    resizeCanvas(canvasRef.current, width, height);
    const context = canvasRef.current.getContext("2d");

    //Our draw come here
    draw(context, state, width, height);
  }, [canvasRef, state, width, height]);

  console.log(state);
  if (state === null) {
    return <div></div>;
  }
  return (
    <div className="game">
      <canvas ref={canvasRef} style={{ position: "fixed" }} />
      <ControlPanel />
    </div>
  );

  // const ports = state.ports.map((port) => <Water />); // TODO: Add port icon
  // const tiles = state.tiles.map((tile, index) => {
  //   // skip desert when placing numbers
  //   const desertIndex = state.tiles.findIndex((t) => t.resource === null);
  //   const numberIndex = index - (desertIndex < index);

  //   return tile.resource === null ? (
  //     <DesertTile />
  //   ) : (
  //     <Tile resource={tile.resource} number={state.numbers[numberIndex]} />
  //   );
  // });
  // const rows = [
  //   [ports[0], <Water />, ports[1], <Water />],
  //   [<Water />, tiles[0], tiles[1], tiles[2], ports[3]],
  //   [ports[8], tiles[3], tiles[4], tiles[5], tiles[6], <Water />],
  //   [<Water />, tiles[7], tiles[8], tiles[9], tiles[10], tiles[11], ports[4]],
  //   [ports[7], tiles[12], tiles[13], tiles[14], tiles[15], <Water />],
  //   [<Water />, tiles[16], tiles[17], tiles[18], ports[5]],
  //   [ports[6], <Water />, ports[7], <Water />],
  // ];

  // return (
  //   <div className="App">
  //     {rows.map((row) => (
  //       <div className="row">{row}</div>
  //     ))}
  //   </div>
  // );
}

export default App;
