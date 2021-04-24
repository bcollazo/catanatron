import React, { useCallback, useState } from "react";
import axios from "axios";
import { useHistory } from "react-router-dom";

import { API_URL } from "../configuration";

import "./HomePage.scss";
import { Button } from "@material-ui/core";
import Loader from "react-loader-spinner";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const history = useHistory();

  const onClick = useCallback(async () => {
    setLoading(true);
    const response = await axios.post(API_URL + "/games");
    const { game_id: gameId } = response.data;
    setLoading(false);
    history.push("/games/" + gameId);
  }, [history]);

  return (
    <div className="home-page">
      <h1 className="logo">Catanatron</h1>
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
