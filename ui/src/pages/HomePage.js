import React, { useCallback, useState } from "react";
import { useHistory } from "react-router-dom";

import "./HomePage.scss";
import { Button } from "@material-ui/core";
import Loader from "react-loader-spinner";
import { createGame } from "../utils/apiClient";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  const onClick = useCallback(async () => {
    setLoading(true);
    const gameId = await createGame();
    setLoading(false);
    history.push("/games/" + gameId);
  }, [history]);

  return (
    <div className="home-page">
      <h1 className="logo">Catanatron</h1>
      <ul>
        <li>1v1 against a bot</li>
        <li>
          open hand <small>(but the bot doesnt use this)</small>
        </li>
        <li>
          no choice during discard <small>(neither does the bot)</small>
        </li>
      </ul>
      {!loading && (
        <Button variant="contained" color="primary" onClick={onClick}>
          Start Game
        </Button>
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
  );
}
