import React, { useCallback, useContext } from "react";
import SwipeableDrawer from "@material-ui/core/SwipeableDrawer";
import Divider from "@material-ui/core/Divider";

import PlayerStateBox from "../components/PlayerStateBox";
import { humanizeAction } from "../components/Prompt";
import { store } from "../store";
import ACTIONS from "../actions";
import { HUMAN_COLOR } from "../constants";

function DrawerContent({ gameState }) {
  const bot =
    gameState && gameState.players.find((x) => x.color !== HUMAN_COLOR);
  const human =
    gameState && gameState.players.find((x) => x.color === HUMAN_COLOR);
  return (
    <>
      <PlayerStateBox
        playerState={bot}
        longestRoad={gameState.longest_roads_by_player[bot.color]}
      />
      <Divider />
      <PlayerStateBox
        playerState={human}
        longestRoad={gameState.longest_roads_by_player[HUMAN_COLOR]}
      />
      <Divider />
      <div className="log">
        {gameState.actions
          .slice()
          .reverse()
          .map((action, i) => (
            <div key={i} className="action">
              {humanizeAction(action)}
            </div>
          ))}
      </div>
    </>
  );
}

export default function LeftDrawer() {
  const { state, dispatch } = useContext(store);
  const iOS = process.browser && /iPad|iPhone|iPod/.test(navigator.userAgent);

  const openLeftDrawer = useCallback(
    (event) => {
      if (
        event &&
        event.type === "keydown" &&
        (event.key === "Tab" || event.key === "Shift")
      ) {
        return;
      }

      dispatch({ type: ACTIONS.SET_LEFT_DRAWER_OPENED, data: true });
    },
    [dispatch]
  );
  const closeLeftDrawer = useCallback(
    (event) => {
      if (
        event &&
        event.type === "keydown" &&
        (event.key === "Tab" || event.key === "Shift")
      ) {
        return;
      }

      dispatch({ type: ACTIONS.SET_LEFT_DRAWER_OPENED, data: false });
    },
    [dispatch]
  );

  return (
    <SwipeableDrawer
      className="left-drawer"
      anchor={"left"}
      open={state.isLeftDrawerOpen}
      onClose={closeLeftDrawer}
      onOpen={openLeftDrawer}
      disableBackdropTransition={!iOS}
      disableDiscovery={iOS}
      onKeyDown={closeLeftDrawer}
    >
      <DrawerContent gameState={state.gameState} />
    </SwipeableDrawer>
  );
}
