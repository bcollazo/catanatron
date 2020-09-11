import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { ControlPanel } from "./ControlPanel";
import { API_URL } from "../configuration";
import SidePanel from "./SidePanel";
import Board from "./Board";

export default function GamePage() {
  const { gameId } = useParams();
  const [state, setState] = useState(null);

  useEffect(() => {
    (async () => {
      const response = await fetch(API_URL + "/games/" + gameId);
      const data = await response.json();
      setState(data);
    })();
  }, [gameId]);

  console.log(state);
  if (state === null) {
    return <div></div>;
  }
  return (
    <div className="h-full flex flex-col">
      <div className="w-full h-full flex">
        <Board state={state} />
        <SidePanel />
      </div>

      <ControlPanel />
    </div>
  );
}
