import React, { useEffect, useState, useCallback, useContext } from "react";
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
import { humanizeAction } from "../components/Prompt";

const ROBOT_THINKING_TIME = 2000;

function GameScreen() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar } = useSnackbar();
  const [inFlightRequest, setInFlightRequest] = useState(false);
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
          enqueueSnackbar(humanizeAction(gameState.actions.slice(-1)[0]));
        }, ROBOT_THINKING_TIME - requestTime);
      })();
    }
  }, [gameId, state.gameState, dispatch, enqueueSnackbar]);

  const onClickNext = useCallback(async () => {
    if (state.gameState && state.gameState.winning_color) {
      return; // do nothing.
    }

    if (inFlightRequest) return; // this makes it idempotent
    setInFlightRequest(true);
    const gameState = await postAction(gameId);
    setInFlightRequest(false);

    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
  }, [gameId, inFlightRequest, setInFlightRequest, state.gameState, dispatch]);

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

  console.log(state.gameState);
  console.log(
    state.gameState.actions.length,
    state.gameState.actions.slice(state.gameState.actions.length - 1),
    state.gameState.current_color,
    state.gameState.current_prompt,
    state.gameState.current_playable_actions
  );
  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard state={state.gameState} />
      <ActionsToolbar onTick={onClickNext} isBotThinking={isBotThinking} />
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
