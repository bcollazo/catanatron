import { useCallback, useContext, useState } from "react";
import SwipeableDrawer from "@mui/material/SwipeableDrawer";
import Divider from "@mui/material/Divider";
import Drawer from "@mui/material/Drawer";
import { CircularProgress, Button } from "@mui/material";
import AssessmentIcon from "@mui/icons-material/Assessment";
import { getMctsAnalysis } from "../utils/apiClient";
import { useParams } from "react-router";

import Hidden from "./Hidden";
import { store } from "../store";
import ACTIONS from "../actions";

import "./RightDrawer.scss";

function DrawerContent() {
  const { gameId } = useParams();
  const { state } = useContext(store);
  const [mctsResults, setMctsResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyzeClick = async () => {
    if (!gameId || !state.gameState || state.gameState.winning_color) return;

    try {
      setLoading(true);
      setError(null);
      const result = await getMctsAnalysis(gameId);
      if (result.success) {
        setMctsResults(result.probabilities);
      } else {
        setError(result.error || "Analysis failed");
      }
    } catch (err) {
      console.error("MCTS Analysis failed:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analysis-box">
      <div className="analysis-header">
        <h3>Win Probability Analysis</h3>
        <Button
          variant="contained"
          color="primary"
          onClick={handleAnalyzeClick}
          disabled={loading || state.gameState?.winning_color}
          startIcon={loading ? <CircularProgress size={20} /> : <AssessmentIcon />}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </Button>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {mctsResults && !loading && !error && (
        <div className="probability-bars">
          {Object.entries(mctsResults).map(([color, probability]) => (
            <div key={color} className={`probability-row ${color.toLowerCase()}`}>
              <span className="player-color">{color}</span>
              <span className="probability-bar">
                <div
                  className="bar-fill"
                  style={{ width: `${probability}%` }}
                />
              </span>
              <span className="probability-value">{probability}%</span>
            </div>
          ))}
        </div>
      )}
      <Divider />
    </div>
  );
}

export default function RightDrawer() {
  const { state, dispatch } = useContext(store);
  const iOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

  const openRightDrawer = useCallback(
    (event) => {
      if (
        event &&
        event.type === "keydown" &&
        (event.key === "Tab" || event.key === "Shift")
      ) {
        return;
      }

      dispatch({ type: ACTIONS.SET_RIGHT_DRAWER_OPENED, data: true });
    },
    [dispatch]
  );

  const closeRightDrawer = useCallback(
    (event) => {
      if (
        event &&
        event.type === "keydown" &&
        (event.key === "Tab" || event.key === "Shift")
      ) {
        return;
      }

      dispatch({ type: ACTIONS.SET_RIGHT_DRAWER_OPENED, data: false });
    },
    [dispatch]
  );

  return (
    <>
      <Hidden breakpoint={{ size: "lg", direction: "up" }} implementation="js">
        <SwipeableDrawer
          className="right-drawer"
          anchor="right"
          open={state.isRightDrawerOpen}
          onClose={closeRightDrawer}
          onOpen={openRightDrawer}
          disableBackdropTransition={!iOS}
          disableDiscovery={iOS}
          onKeyDown={closeRightDrawer}
        >
          <DrawerContent />
        </SwipeableDrawer>
      </Hidden>
      <Hidden breakpoint={{ size: "md", direction: "down" }} implementation="css">
        <Drawer
          className="right-drawer"
          anchor="right"
          variant="permanent"
          open
        >
          <DrawerContent />
        </Drawer>
      </Hidden>
    </>
  );
}
