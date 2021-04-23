import React, { useCallback, useState } from "react";
import axios from "axios";
import { useHistory } from "react-router-dom";

import { API_URL } from "../configuration";

import "./HomePage.scss";
import { Button } from "@material-ui/core";

export default function HomePage() {
  const [disabled, setDisabled] = useState(false);
  const history = useHistory();

  const onClick = useCallback(async () => {
    setDisabled(true);
    const response = await axios.post(API_URL + "/games");
    const { game_id: gameId } = response.data;
    setDisabled(false);
    history.push("/games/" + gameId);
  }, [history]);

  return (
    <div className="home-page">
      <h1>Catanatron</h1>
      <Button
        disabled={disabled}
        variant="contained"
        color="primary"
        onClick={onClick}
      >
        Start Game
      </Button>
    </div>
  );
}
