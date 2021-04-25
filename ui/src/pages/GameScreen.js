import React, { useEffect, useState, useCallback, useContext } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";

import { API_URL } from "../configuration";
import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./GameScreen.scss";
import { BOT_COLOR } from "../constants";
import LeftDrawer from "../components/LeftDrawer";
import { store } from "../store";
import ACTIONS from "../actions";

function GameScreen() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const [inFlightRequest, setInFlightRequest] = useState(false);
  const [isBotThinking, setIsBotThinking] = useState(false);

  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const response = await axios.get(API_URL + "/games/" + gameId);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: response.data });
    })();
  }, [gameId, dispatch]);

  useEffect(() => {
    if (!state.gameState) {
      return;
    }

    // Kick off next query?
    if (state.gameState.current_color === BOT_COLOR) {
      (async () => {
        console.log("kicking off, thinking-play cycle");
        setIsBotThinking(true);
        const response = await axios.post(`${API_URL}/games/${gameId}/actions`);
        setTimeout(() => {
          // simulate thinking
          setIsBotThinking(false);
          dispatch({ type: ACTIONS.SET_GAME_STATE, data: response.data });
        }, 2000);
      })();
    }
  }, [gameId, state.gameState, dispatch]);

  const onClickNext = useCallback(async () => {
    if (state.gameState && state.gameState.winning_color) {
      return; // do nothing.
    }

    if (inFlightRequest) return; // this makes it idempotent
    setInFlightRequest(true);
    const response = await axios.post(`${API_URL}/games/${gameId}/actions`);
    setInFlightRequest(false);

    dispatch({ type: ACTIONS.SET_GAME_STATE, data: response.data });
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
