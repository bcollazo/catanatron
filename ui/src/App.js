import React, { useEffect, useState } from "react";
import cn from "classnames";

import "./App.scss";

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

function App() {
  const [state, setState] = useState(null);

  useEffect(() => {
    (async () => {
      const response = await fetch(URL);
      const data = await response.json();
      setState(data);
    })();
  }, []);

  if (state === null) {
    return <div></div>;
  }

  const ports = state.ports.map((port) => <Water />); // TODO: Add port icon
  const tiles = state.tiles.map((tile, index) => {
    // skip desert when placing numbers
    const desertIndex = state.tiles.findIndex((t) => t.resource === null);
    const numberIndex = index - (desertIndex < index);

    return tile.resource === null ? (
      <DesertTile />
    ) : (
      <Tile resource={tile.resource} number={state.numbers[numberIndex]} />
    );
  });
  const rows = [
    [ports[0], <Water />, ports[1], <Water />],
    [<Water />, tiles[0], tiles[1], tiles[2], ports[3]],
    [ports[8], tiles[3], tiles[4], tiles[5], tiles[6], <Water />],
    [<Water />, tiles[7], tiles[8], tiles[9], tiles[10], tiles[11], ports[4]],
    [ports[7], tiles[12], tiles[13], tiles[14], tiles[15], <Water />],
    [<Water />, tiles[16], tiles[17], tiles[18], ports[5]],
    [ports[6], <Water />, ports[7], <Water />],
  ];

  return (
    <div className="App">
      {rows.map((row) => (
        <div className="row">{row}</div>
      ))}
    </div>
  );
}

export default App;
