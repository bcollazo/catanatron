import { useEffect, useContext, useState } from "react";
import { useParams } from "react-router-dom";
import { GridLoader } from "react-spinners";

import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

// import "./ReplayScreen.scss";
import LeftDrawer from "../components/LeftDrawer";
import RightDrawer from "../components/RightDrawer";
import { store } from "../store";
import ACTIONS from "../actions";
import { type StateIndex, getState } from "../utils/apiClient";
import AnalysisBox from "../components/AnalysisBox";
import { Divider } from "@mui/material";

function ReplayScreen() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const [stateIndex, setStateIndex] = useState<number>(5);

  const handlePrevState = () => setStateIndex((prev) => Math.max(prev - 1, 0));
  const handleNextState = () => setStateIndex((prev) => prev + 1);

  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const gameState = await getState(gameId, stateIndex as StateIndex);
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
      <ActionsToolbar isBotThinking={false} replayMode={true} />
      <LeftDrawer />
      <RightDrawer>
        <AnalysisBox />
        <Divider />
      </RightDrawer>
    </main>
  );
}

export default ReplayScreen;
