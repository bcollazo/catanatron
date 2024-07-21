import React, { useCallback, useContext, useEffect, useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import memoize from "fast-memoize";
import { useMediaQuery, useTheme } from "@material-ui/core";

import useWindowSize from "../utils/useWindowSize";

import "./Board.scss";
import { store } from "../store";
import { isPlayersTurn } from "../utils/stateUtils";
import { postAction } from "../utils/apiClient";
import { useParams } from "react-router";
import ACTIONS from "../actions";
import Board from "./Board";

/**
 * Returns object representing actions to be taken if click on node.
 * @returns {3 => ["BLUE", "BUILD_CITY", 3], ...}
 */
function buildNodeActions(state) {
  if (!isPlayersTurn(state.gameState)) {
    return {};
  }

  const nodeActions = {};
  const buildInitialSettlementActions = state.gameState.is_initial_build_phase
    ? state.gameState.current_playable_actions.filter(
        (action) => action[1] === "BUILD_SETTLEMENT"
      )
    : [];
  const inInitialBuildPhase = state.gameState.is_initial_build_phase;
  if (inInitialBuildPhase) {
    buildInitialSettlementActions.forEach((action) => {
      nodeActions[action[2]] = action;
    });
  } else if (state.isBuildingSettlement) {
    state.gameState.current_playable_actions
      .filter((action) => action[1] === "BUILD_SETTLEMENT")
      .forEach((action) => {
        nodeActions[action[2]] = action;
      });
  } else if (state.isBuildingCity) {
    state.gameState.current_playable_actions
      .filter((action) => action[1] === "BUILD_CITY")
      .forEach((action) => {
        nodeActions[action[2]] = action;
      });
  }
  return nodeActions;
}

function buildEdgeActions(state) {
  if (!isPlayersTurn(state.gameState)) {
    return {};
  }

  const edgeActions = {};
  const buildInitialRoadActions = state.gameState.is_initial_build_phase
    ? state.gameState.current_playable_actions.filter(
        (action) => action[1] === "BUILD_ROAD"
      )
    : [];
  const inInitialBuildPhase = state.gameState.is_initial_build_phase;
  if (inInitialBuildPhase) {
    buildInitialRoadActions.forEach((action) => {
      edgeActions[action[2]] = action;
    });
  } else if (state.isBuildingRoad || state.isRoadBuilding) {
    state.gameState.current_playable_actions
      .filter((action) => action[1] === "BUILD_ROAD")
      .forEach((action) => {
        edgeActions[action[2]] = action;
      });
  }
  return edgeActions;
}

export default function ZoomableBoard({ replayMode }) {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const { width, height } = useWindowSize();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.up("md"));
  const [show, setShow] = useState(false);

  // TODO: Move these up to GameScreen and let Zoomable be presentational component
  // https://stackoverflow.com/questions/61255053/react-usecallback-with-parameter
  const buildOnNodeClick = useCallback(
    memoize((id, action) => async () => {
      console.log("Clicked Node ", id, action);
      if (action) {
        const gameState = await postAction(gameId, action);
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      }
    }),
    []
  );
  const buildOnEdgeClick = useCallback(
    memoize((id, action) => async () => {
      console.log("Clicked Edge ", id, action);
      if (action) {
        const gameState = await postAction(gameId, action);
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      }
    }),
    []
  );
  const handleTileClick = useCallback(
    memoize((coordinate) => {
      console.log("Clicked Tile ", coordinate);
      if (state.isMovingRobber) {
        // Find the "MOVE_ROBBER" action in current_playable_actions that 
        // corresponds to the tile coordinate selected by the user
        const matchingAction = state.gameState.current_playable_actions.find(
          ([, action_type, [action_coordinate, ,]]) =>
            action_type === "MOVE_ROBBER" &&
            action_coordinate.every((val, index) => val === coordinate[index])
        );
        if (matchingAction) {
          postAction(gameId, matchingAction)
            .then(gameState => {
              dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
            });
        }
      }
    }),
    [state.isMovingRobber]
  );

  const nodeActions = replayMode ? {} : buildNodeActions(state);
  const edgeActions = replayMode ? {} : buildEdgeActions(state);

  useEffect(() => {
    setTimeout(() => {
      setShow(true);
    }, 300);
  }, []);

  return (
    <TransformWrapper
      options={{
        limitToBounds: true,
      }}
    >
      {({
        zoomIn,
        zoomOut,
        resetTransform,
        positionX,
        positionY,
        scale,
        previousScale,
      }) => (
        <React.Fragment>
          <div className="board-container">
            <TransformComponent>
              <Board
                width={width}
                height={height}
                buildOnNodeClick={buildOnNodeClick}
                buildOnEdgeClick={buildOnEdgeClick}
                handleTileClick={handleTileClick}
                nodeActions={nodeActions}
                edgeActions={edgeActions}
                replayMode={replayMode}
                show={show}
                gameState={state.gameState}
                isMobile={isMobile}
                isMovingRobber={state.isMovingRobber}
              ></Board>
            </TransformComponent>
          </div>
        </React.Fragment>
      )}
    </TransformWrapper>
  );
}
