import React, { useEffect, useState, useRef } from "react";
import cn from "classnames";

import "./App.scss";
import useWindowSize from "./useWindowSize";

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

const drawTile = (ctx, x, y, radius, color, number) => {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fill();

  if (number !== undefined) {
    ctx.font = "30px Helvetica";
    ctx.fillStyle = "white";
    ctx.textAlign = "center";
    ctx.fillText(number, x - 1, y + 8);
  }
};

const color = (tile) => {
  if (tile.type === "WATER" || tile.type === "PORT") {
    return "#0000ff";
  } else if (tile.type === "DESERT") {
    return "orange";
  } else {
    console.log(tile.resource);
    return {
      SHEEP: "lightgreen",
      WOOD: "green",
      BRICK: "red",
      ORE: "gray",
      WHEAT: "yellow",
    }[tile.resource];
  }
};

const draw = (ctx, state, windowWidth, windowHeight) => {
  const radius = 40;
  const diameter = 2 * radius;
  const tileHeight = 60;

  const centerX = windowWidth / 2 - radius;
  const centerY = windowHeight / 2 - radius;
  for (const { coordinate, tile } of state.tiles) {
    // add 2 diameters on 60deg for each X (0.5 on x and 0.86602540378 on y)
    // add 2 diameters on 120deg for each Y (-0.5 on x and 0.86602540378 on y)
    const [x, y, z] = coordinate;
    const canvasX = centerX + x * 2 * diameter * 0.5 + y * 2 * diameter * -0.5;
    const canvasY =
      centerY +
      z * diameter +
      x * 2 * diameter * 0.86 +
      y * 2 * diameter * 0.86;
    drawTile(ctx, canvasX, canvasY, radius, color(tile), tile.number);
    console.log(coordinate);
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
  return <canvas ref={canvasRef} style={{ position: "fixed" }} />;

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
