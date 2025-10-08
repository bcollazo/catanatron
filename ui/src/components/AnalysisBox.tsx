import { useContext, useState } from "react";
import { CircularProgress, Button } from "@mui/material";
import AssessmentIcon from "@mui/icons-material/Assessment";
import { type MCTSProbabilities, type StateIndex, getMctsAnalysis } from "../utils/apiClient";
import { useParams } from "react-router";

import "./AnalysisBox.scss";
import { store } from "../store";

type AnalysisBoxProps = {
    stateIndex: StateIndex;
}

export default function AnalysisBox( { stateIndex }: AnalysisBoxProps ) {
  const { gameId } = useParams();
  const { state } = useContext(store);
  const [mctsResults, setMctsResults] = useState<MCTSProbabilities | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyzeClick = async () => {
    if (!gameId || !state.gameState || state.gameState.winning_color) return;

    try {
      setLoading(true);
      setError('');
      const result = await getMctsAnalysis(gameId, stateIndex);
      if (result.success) {
        setMctsResults(result.probabilities);
      } else {
        setError(result.error || "Analysis failed");
      }
    } catch (err) {
      console.error("MCTS Analysis failed:", err);
      if (err instanceof Error) {
        setError(err.message);
      } else if (typeof err === "string") {
        setError(err);
      } else {
        setError("An unknown error occurred");
      }
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
          disabled={loading || !!state.gameState?.winning_color}
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
    </div>
  );
}