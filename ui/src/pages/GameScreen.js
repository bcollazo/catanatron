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

function GameScreen(props) {
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

  return (
    <main>
      <div className="prompt">
        {state && `${state.current_color}: ${state.current_prompt}`}
      </div>
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
