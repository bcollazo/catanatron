import React, { useContext, useEffect, useState } from "react";
import { Drawer, Hidden, SwipeableDrawer, CircularProgress } from "@material-ui/core";
import { store } from "../store";
import ACTIONS from "../actions";
import { postMctsAnalysis } from "../utils/apiClient";
import { getHumanColor, isPlayersTurn } from "../utils/stateUtils";

import "./RightDrawer.scss";

function AnalysisContent({ gameState }) {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const humanColor = getHumanColor(gameState);
  const isHumanTurn = isPlayersTurn(gameState);

  useEffect(() => {
    // Reset analysis when game state changes
    setAnalysisResult(null);
    
    // Only analyze when it's human's turn and valid moves exist
    if (isHumanTurn && gameState.current_playable_actions.length > 0) {
      setIsLoading(true);
      postMctsAnalysis(gameState)
        .then(result => {
          setAnalysisResult(result);
          setIsLoading(false);
        })
        .catch(error => {
          console.error("MCTS Analysis failed:", error);
          setIsLoading(false);
        });
    }
  }, [gameState, isHumanTurn]);

  return (
    <div className="analysis-box">
      <div className="analysis-header">
        <h3>Move Analysis</h3>
      </div>
      
      {isLoading ? (
        <div className="loading-indicator">
          <CircularProgress size={24} />
          <p>Analyzing moves...</p>
        </div>
      ) : analysisResult ? (
        <div className="probability-bars">
          {Object.entries(analysisResult).map(([color, probability]) => (
            <div key={color} className={`probability-row ${color.toLowerCase()}`}>
              <span>{color}</span>
              <span className="probability-value">
                {(probability * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-analysis">
          {isHumanTurn ? "Waiting for analysis..." : "Waiting for your turn..."}
        </div>
      )}
    </div>
  );
}

export default function RightDrawer() {
  const { state, dispatch } = useContext(store);
  const iOS = process.browser && /iPad|iPhone|iPod/.test(navigator.userAgent);

  const closeRightDrawer = () => {
    dispatch({ type: ACTIONS.SET_RIGHT_DRAWER_OPENED, data: false });
  };

  const openRightDrawer = () => {
    dispatch({ type: ACTIONS.SET_RIGHT_DRAWER_OPENED, data: true });
  };

  return (
    <>
      <Hidden lgUp implementation="js">
        <SwipeableDrawer
          className="right-drawer"
          anchor="right"
          open={state.isRightDrawerOpen}
          onClose={closeRightDrawer}
          onOpen={openRightDrawer}
          disableBackdropTransition={!iOS}
          disableDiscovery={iOS}
        >
          <AnalysisContent gameState={state.gameState} />
        </SwipeableDrawer>
      </Hidden>
      <Hidden mdDown implementation="css">
        <Drawer
          className="right-drawer"
          anchor="right"
          variant="permanent"
          open
        >
          <AnalysisContent gameState={state.gameState} />
        </Drawer>
      </Hidden>
    </>
  );
}