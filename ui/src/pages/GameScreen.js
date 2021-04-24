import React, { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";
import SwipeableDrawer from "@material-ui/core/SwipeableDrawer";
import Divider from "@material-ui/core/Divider";

import { API_URL } from "../configuration";
import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";
import PlayerStateBox from "../components/PlayerStateBox";
import { humanizeAction } from "../components/Prompt";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./GameScreen.scss";
import { BOT_COLOR, HUMAN_COLOR } from "../constants";

function DrawerContent({ state, bot, human }) {
  return (
    <>
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
      <div className="log">
        {state.actions.map((a) => (
          <div className="action">{humanizeAction(a)}</div>
        ))}
      </div>
    </>
    // <div
    //   className={classes.list}
    //   role="presentation"

    // >

    // </div>
  );
}
function GameScreen() {
  const { gameId } = useParams();
  const [state, setState] = useState(null);
  const [inFlightRequest, setInFlightRequest] = useState(false);
  const [isBotThinking, setIsBotThinking] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const response = await axios.get(API_URL + "/games/" + gameId);
      setState(response.data);
    })();
  }, [gameId]);

  useEffect(() => {
    if (!state) {
      return;
    }

    // Kick off next query?
    if (state.current_color === BOT_COLOR) {
      (async () => {
        console.log("kicking off, thinking-play cycle");
        setIsBotThinking(true);
        const response = await axios.post(`${API_URL}/games/${gameId}/actions`);
        setTimeout(() => {
          // simulate thinking
          setIsBotThinking(false);
          setState(response.data);
        }, 2000);
      })();
    }
  }, [gameId, state]);

  const onClickNext = useCallback(async () => {
    if (state && state.winning_color) {
      return; // do nothing.
    }

    if (inFlightRequest) return; // this makes it idempotent
    setInFlightRequest(true);
    const response = await axios.post(`${API_URL}/games/${gameId}/actions`);
    setInFlightRequest(false);

    setState(response.data);
  }, [gameId, inFlightRequest, setInFlightRequest, state]);

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

  const iOS = process.browser && /iPad|iPhone|iPod/.test(navigator.userAgent);

  const toggleLeftDrawer = (open) => (event) => {
    if (
      event &&
      event.type === "keydown" &&
      (event.key === "Tab" || event.key === "Shift")
    ) {
      return;
    }

    setIsDrawerOpen(open);
  };

  console.log(state);
  console.log(
    state.actions.length,
    state.actions.slice(state.actions.length - 1),
    state.current_color,
    state.current_prompt
  );
  const gameOver = state && state.winning_color;
  const bot = state && state.players.find((x) => x.color !== HUMAN_COLOR);
  const human = state && state.players.find((x) => x.color === HUMAN_COLOR);
  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard state={state} />
      <ActionsToolbar
        onTick={onClickNext}
        disabled={gameOver || inFlightRequest}
        toggleLeftDrawer={toggleLeftDrawer}
        state={state}
        isBotThinking={isBotThinking}
      />
      <SwipeableDrawer
        className="left-drawer"
        anchor={"left"}
        open={isDrawerOpen}
        onClose={toggleLeftDrawer(false)}
        onOpen={toggleLeftDrawer(true)}
        disableBackdropTransition={!iOS}
        disableDiscovery={iOS}
        onKeyDown={toggleLeftDrawer(false)}
      >
        <DrawerContent
          toggleDrawer={toggleLeftDrawer}
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
