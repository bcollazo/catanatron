import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";
import SwipeableDrawer from "@material-ui/core/SwipeableDrawer";

import Divider from "@material-ui/core/Divider";
import { makeStyles } from "@material-ui/core/styles";

import { API_URL } from "../configuration";
import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./GameScreen.scss";
import PlayerStateBox, { ResourceCards } from "../components/PlayerStateBox";
import Prompt from "../components/Prompt";

const HUMAN_COLOR = "BLUE";

const useStyles = makeStyles({
  list: {
    width: 250,
  },
});

function DrawerContent({ toggleDrawer, state, bot, human }) {
  const classes = useStyles();

  return (
    <div
      className={classes.list}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <PlayerStateBox
        playerState={bot}
        longestRoad={state.longest_roads_by_player[bot.color]}
      />
      <Divider />
      <PlayerStateBox
        playerState={human}
        longestRoad={state.longest_roads_by_player[HUMAN_COLOR]}
      />
      <Divider />
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
  return actions.slice(i, numActions);
}

function GameScreen() {
  const { gameId } = useParams();
  const [actionQueue, setActionQueue] = useState([]);
  const [state, setState] = useState(null);
  const [inFlightRequest, setInFlightRequest] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const iOS = process.browser && /iPad|iPhone|iPod/.test(navigator.userAgent);

  const toggleDrawer = (open) => (event) => {
    if (
      event &&
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }

    setIsDrawerOpen(open);
  };

  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const response = await axios.get(API_URL + "/games/" + gameId);
      const queue = getQueue(response.data.actions);
      setActionQueue(queue);
      setState(response.data);
    })();
  }, [gameId]);

  const onClickNext = useCallback(async () => {
    if (state && state.winning_color) {
      return; // do nothing.
    }

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
  }, [gameId, inFlightRequest, setInFlightRequest, actionQueue, state]);

  if (!state) {
    return (
      <Loader
        className="loader"
        type="Grid"
        color="#000000"
        height={100}
        width={100}
      />
    );
  }

  const gameOver = state && state.winning_color;
  const bot = state && state.players.find((x) => x.color !== HUMAN_COLOR);
  const human = state && state.players.find((x) => x.color === HUMAN_COLOR);
  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard state={state} />
      <ResourceCards playerState={human} />
      <Prompt actionQueue={actionQueue} state={state} />
      <ActionsToolbar
        onTick={onClickNext}
        disabled={gameOver || inFlightRequest}
        botsTurn={actionQueue.length !== 0 && actionQueue[0] !== HUMAN_COLOR}
        prompt={state.current_prompt}
      />
      <SwipeableDrawer
        anchor={"left"}
        open={isDrawerOpen}
        onClose={toggleDrawer(false)}
        onOpen={toggleDrawer(true)}
        disableBackdropTransition={!iOS}
        disableDiscovery={iOS}
      >
        <DrawerContent
          toggleDrawer={toggleDrawer}
          state={state}
          human={human}
          bot={bot}
        />
      </SwipeableDrawer>
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
