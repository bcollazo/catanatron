import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";

import { API_URL } from "../configuration";
import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./GameScreen.scss";

const HUMAN_COLOR = "BLUE";

function Prompt({ actionQueue, state }) {
  let prompt = "";
  if (actionQueue.length === 0) {
    prompt = `${state.current_color}: ${state.current_prompt}`;
  } else {
    prompt = `${actionQueue[0][0]}: ${actionQueue[0].slice(1)}`;
  }
  return <div className="prompt">{state && prompt}</div>;
}

function getQueue(actions) {
  const numActions = actions.length;
  let i;
  for (i = numActions - 1; i >= 0; i--) {
    if (actions[i][0] === HUMAN_COLOR) {
      break;
    }
  }
  i++;
  console.log(i, actions.slice(i, numActions));
  return actions.slice(i, numActions);
}

function GameScreen() {
  const { gameId } = useParams();
  const [actionQueue, setActionQueue] = useState([]);
  const [state, setState] = useState(null);
  const [inFlightRequest, setInFlightRequest] = useState(false);

  useEffect(() => {
    (async () => {
      const response = await axios.get(API_URL + "/games/" + gameId);

      const queue = getQueue(response.data.actions);
      setActionQueue(queue);
      setState(response.data);
    })();
  }, [gameId]);

  const onClickNext = useCallback(async () => {
    // If you queue, consume from queue, else populate
    if (actionQueue.length > 0) {
      setActionQueue(actionQueue.slice(1));
    } else {
      if (inFlightRequest) return; // this makes it idempotent
      setInFlightRequest(true);
      const response = await axios.post(`${API_URL}/games/${gameId}/actions`);
      setInFlightRequest(false);

      const queue = getQueue(response.data.actions);
      setActionQueue(queue);
      setState(response.data);
    }
  }, [gameId, inFlightRequest, setInFlightRequest, actionQueue]);

  console.log(state);
  return (
    <main>
      {state && <Prompt actionQueue={actionQueue} state={state} />}
      {!state && (
        <Loader
          className="loader"
          type="Grid"
          color="#FFFFFF"
          height={100}
          width={100}
        />
      )}
      {state && <ZoomableBoard state={state} />}
      <ActionsToolbar onTick={onClickNext} />
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
