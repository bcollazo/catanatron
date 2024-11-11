import React, { useState } from "react";
import { useHistory } from "react-router-dom";
import { Button } from "@material-ui/core";
import Loader from "react-loader-spinner";
import { createGame } from "../utils/apiClient";

import "./HomePage.scss";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [numPlayers, setNumPlayers] = useState(2);
  const history = useHistory();

  const handleCreateGame = async (getPlayers) => {
    setLoading(true);
    const players = getPlayers(numPlayers);
    const gameId = await createGame(players);
    setLoading(false);
    history.push("/games/" + gameId);
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
              onClick={() => {
                handleCreateGame((numPlayers) => {
                  const players = ["HUMAN"];
                  for (let i = 1; i < numPlayers; i++) {
                    players.push("CATANATRON");
                  }
                  return players;
                });
              }}
            >
              Play against Catanatron
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => {
                handleCreateGame((numPlayers) => {
                  const players = Array(numPlayers).fill("RANDOM");
                  return players;
                });
              }}
            >
              Watch Random Bots
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => {
                handleCreateGame((numPlayers) => {
                  const players = Array(numPlayers).fill("CATANATRON");
                  return players;
                });
              }}
            >
              Watch Catanatron
            </Button>
          </>
        ) : (
          <Loader
            className="loader"
            type="Grid"
            color="#ffffff"
            height={60}
            width={60}
          />
        )}
      </div>
    </div>
  );
}
