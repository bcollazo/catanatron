import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

import { ControlPanel } from "./ControlPanel";
import { API_URL } from "../configuration";
import SidePanel from "./SidePanel";
import Board from "./Board";

export default function GamePage() {
  const { gameId } = useParams();
  const [state, setState] = useState(null);
  const [automation, setAutomation] = useState(false);
  const [inFlightRequest, setInFlightRequest] = useState(false);

  useEffect(() => {
    (async () => {
      const response = await fetch(API_URL + "/games/" + gameId);
      const data = await response.json();
      setState(data);
    })();
  }, [gameId]);

  const onClickNext = useCallback(async () => {
    setInFlightRequest(true);
    const response = await axios.post(`${API_URL}/games/${gameId}/tick`);
    setInFlightRequest(false);
    setState(response.data);
  }, [gameId]);

  const onClickAutomation = () => {
    setAutomation(!automation);
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      if (automation && !inFlightRequest) {
        await onClickNext();
      }
    }, 50);
    return () => clearInterval(interval);
  }, [automation, inFlightRequest, onClickNext]);

  console.log(state);
  if (state === null) {
    return <div></div>;
  }
  return (
    <div className="h-full flex">
      <div
        className="w-full h-full flex flex-col"
        style={{ backgroundColor: "#01c7f4" }}
      >
        <Board state={state} />
        <ControlPanel
          onClickNext={onClickNext}
          onClickAutomation={onClickAutomation}
        />
      </div>
      <SidePanel state={state} />
    </div>
  );
}
