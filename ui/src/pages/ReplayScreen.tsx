import { useEffect, useContext, useState } from "react";
import { useParams } from "react-router-dom";
import { GridLoader } from "react-spinners";

import ZoomableBoard from "./ZoomableBoard";

import "./ReplayScreen.scss";
import LeftDrawer from "../components/LeftDrawer";
import RightDrawer from "../components/RightDrawer";
import { store } from "../store";
import ACTIONS from "../actions";
import { getState } from "../utils/apiClient";
import AnalysisBox from "../components/AnalysisBox";
import { Divider } from "@mui/material";
import ReplayBox from "../components/ReplayBox";

function ReplayScreen() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const [latestStateIndex, setLatestStateIndex] = useState<number>(0);
  const [stateIndex, setStateIndex] = useState<number>(0);

  const handlePrevState = () => setStateIndex((prev) => Math.max(prev - 1, 0));
  const handleNextState = () => setStateIndex((prev) => Math.min(prev + 1, latestStateIndex));

  useEffect(() => {
    if (!gameId) return;

    (async () => {
      const latestState = await getState(gameId, "latest");
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: latestState });
      setLatestStateIndex(latestState.state_index);
    })();
  }, [gameId, dispatch]);

  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const gameState = await getState(gameId, stateIndex);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    })();
  }, [gameId, stateIndex, dispatch]);

  if (!state.gameState) {
    return (
      <main>
        <GridLoader
          className="loader"
          color="#000000"
          size={100}
        />
      </main>
    );
  }

  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard replayMode={true} />
      <LeftDrawer />
      <RightDrawer>
        <AnalysisBox stateIndex={stateIndex}/>
        <Divider />
        <ReplayBox
          stateIndex={stateIndex}
          latestStateIndex={latestStateIndex}
          onNextMove={handleNextState}
          onPrevMove={handlePrevState}
          onSeekMove={(index) => setStateIndex(index)}
        />
      </RightDrawer>
    </main>
  );
}

export default ReplayScreen;
