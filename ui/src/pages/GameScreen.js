import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";
import cn from "classnames";

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

function PlayerStateBox({ playerState, longestRoad }) {
  const numDevCards = Object.values(playerState.development_deck).reduce(
    (a, b) => a + b,
    0
  );
  const resdeck = playerState.resource_deck;
  return (
    <div className="player-state-box">
      <div className="resource-cards" title="Resource Cards">
        {resdeck.WOOD !== 0 && (
          <div className="wood-cards center-text">{resdeck.WOOD}</div>
        )}
        {resdeck.BRICK !== 0 && (
          <div className="brick-cards center-text">{resdeck.BRICK}</div>
        )}
        {resdeck.SHEEP !== 0 && (
          <div className="sheep-cards center-text">{resdeck.SHEEP}</div>
        )}
        {resdeck.WHEAT !== 0 && (
          <div className="wheat-cards center-text">{resdeck.WHEAT}</div>
        )}
        {resdeck.ORE !== 0 && (
          <div className="ore-cards center-text">{resdeck.ORE}</div>
        )}
        {numDevCards !== 0 && (
          <div className="dev-cards center-text" title="Development Cards">
            {numDevCards}
          </div>
        )}
      </div>
      <div
        className={cn("num-knights center-text", {
          has_army: playerState.has_army,
        })}
        title="Knights Played"
      >
        <span>{playerState.played_development_cards.KNIGHT}</span>
        <small>knights</small>
      </div>
      <div
        className={cn("num-roads center-text", {
          has_road: playerState.has_road,
        })}
        title="Longest Road"
      >
        {longestRoad}
        <small>roads</small>
      </div>
      <div
        className={cn("victory-points center-text", {
          has_road: playerState.actual_victory_points >= 10,
        })}
        title="Victory Points"
      >
        {playerState.actual_victory_points}
        <small>VPs</small>
      </div>
    </div>
  );
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
  const bot = state && state.players.find((x) => x.color !== HUMAN_COLOR);
  const human = state && state.players.find((x) => x.color === HUMAN_COLOR);
  return (
    <main>
      {state && <Prompt actionQueue={actionQueue} state={state} />}
      {state && (
        <PlayerStateBox
          playerState={bot}
          longestRoad={state.longest_roads_by_player[bot.color]}
        />
      )}
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
      {state && (
        <PlayerStateBox
          playerState={human}
          longestRoad={state.longest_roads_by_player[HUMAN_COLOR]}
        />
      )}
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
