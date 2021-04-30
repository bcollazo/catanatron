import React, { useEffect, useState, useContext } from "react";
import { useParams } from "react-router-dom";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";
import { useSnackbar } from "notistack";

import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./GameScreen.scss";
import { BOT_COLOR } from "../constants";
import LeftDrawer from "../components/LeftDrawer";
import { store } from "../store";
import ACTIONS from "../actions";
import { getState, postAction } from "../utils/apiClient";
import { dispatchSnackbar } from "../components/Snackbar";

const ROBOT_THINKING_TIME = 2000;

function GameScreen() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const [isBotThinking, setIsBotThinking] = useState(false);

  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const gameState = await getState(gameId);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    })();
  }, [gameId, dispatch]);

  useEffect(() => {
    if (!state.gameState) {
      return;
    }

    // Kick off next query?
    if (state.gameState.current_color === BOT_COLOR) {
      (async () => {
        setIsBotThinking(true);
        const start = new Date();
        const gameState = await postAction(gameId);
        const requestTime = new Date() - start;
        setTimeout(() => {
          // simulate thinking
          setIsBotThinking(false);
          dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
          dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
        }, ROBOT_THINKING_TIME - requestTime);
      })();
    }
  }, [gameId, state.gameState, dispatch, enqueueSnackbar, closeSnackbar]);

  if (!state.gameState) {
    return (
      <main>
        <Loader
          className="loader"
          type="Grid"
          color="#000000"
          height={100}
          width={100}
        />
      </main>
    );
  }

  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard state={state.gameState} />
      <ActionsToolbar isBotThinking={isBotThinking} />
      <LeftDrawer />
    </main>
  );
}

GameScreen.propTypes = {
  /**
   * Injected by the documentation to work in an iframe.
   * You won't need it on your project.
   */
  window: PropTypes.func,
};

export default GameScreen;
