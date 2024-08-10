import React, { useState } from "react";
import { useHistory } from "react-router-dom";

import "./HomePage.scss";
import { Button } from "@material-ui/core";
import Loader from "react-loader-spinner";
import { createGame } from "../utils/apiClient";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  const handleCreateGame = async (players) => {
    setLoading(true);
    const gameId = await createGame(players);
    setLoading(false);
    history.push("/games/" + gameId);
  };

  return (
    <div className="home-page">
      <h1 className="logo">Catanatron</h1>
      <div className="switchable">
        {!loading && (
          <>
            <ul>
              <li>1V1</li>
              <li>OPEN HAND</li>
              <li>NO CHOICE DURING DISCARD</li>
            </ul>
            <Button
              variant="contained"
              color="primary"
              onClick={() => handleCreateGame(["HUMAN", "CATANATRON"])}
            >
              Play against Catanatron
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(["RANDOM", "RANDOM"])}
            >
              Watch Random Bots
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(["MyPlayer", "RANDOM"])}
            >
              Watch PPO v Random
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={() => handleCreateGame(["CATANATRON", "CATANATRON"])}
            >
              Watch Catanatron
            </Button>
          </>
        )}
        {loading && (
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
