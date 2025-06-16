import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@mui/material";
import { GridLoader } from "react-spinners";
import { createGame } from "../utils/apiClient";

import "./HomePage.scss";

// Enum of Type of Game Mode
const GameMode = Object.freeze({
  HUMAN_VS_CATANATRON: "HUMAN_VS_CATANATRON",
  RANDOM_BOTS: "RANDOM_BOTS",
  CATANATRON_BOTS: "CATANATRON_BOTS",
});

type GameModeType = typeof GameMode[keyof typeof GameMode]

function getPlayers(gameMode: GameModeType, numPlayers: number) {
  switch (gameMode) {
    case GameMode.HUMAN_VS_CATANATRON:
      const players = ["HUMAN"];
      for (let i = 1; i < numPlayers; i++) {
        players.push("CATANATRON");
      }
      return players;
    case GameMode.RANDOM_BOTS:
      return Array(numPlayers).fill("RANDOM");
    case GameMode.CATANATRON_BOTS:
      return Array(numPlayers).fill("CATANATRON");
    default:
      throw new Error("Invalid Game Mode");
  }
}

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [numPlayers, setNumPlayers] = useState(2);
  const navigate = useNavigate();

  const handleCreateGame = async (gameMode: GameModeType) => {
    setLoading(true);
    const players = getPlayers(gameMode, numPlayers);
    const gameId = await createGame(players);
    setLoading(false);
    navigate("/games/" + gameId);
  };

  return (
    <div className="home-page">
      <h1 className="logo">Catanatron</h1>

      <div className="switchable">
        {!loading ? (
          <>
            <ul>
              <li>OPEN HAND</li>
              <li>NO CHOICE DURING DISCARD</li>
            </ul>
            <div className="player-count-selector">
              <div className="player-count-label">Number of Players</div>
              <div className="player-count-buttons">
                {[2, 3, 4].map((value) => (
                  <Button
                    key={value}
                    variant="contained"
                    onClick={() => setNumPlayers(value)}
                    className={`player-count-button ${
                      numPlayers === value ? "selected" : ""
                    }`}
                  >
                    {value} Players
                  </Button>
                ))}
              </div>
            </div>
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleCreateGame(GameMode.HUMAN_VS_CATANATRON)}
            >
              Play against Catanatron
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(GameMode.RANDOM_BOTS)}
            >
              Watch Random Bots
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(GameMode.CATANATRON_BOTS)}
            >
              Watch Catanatron
            </Button>
          </>
        ) : (
          <GridLoader
            className="loader"
            color="#ffffff"
            size={60}
          />
        )}
      </div>
    </div>
  );
}
