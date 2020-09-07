import React, { useCallback, useState } from "react";
import axios from "axios";
import { useHistory } from "react-router-dom";

import "./HomePage.scss";
import { API_URL } from "../configuration";

export default function HomePage() {
  const [disabled, setDisabled] = useState(false);
  const history = useHistory();

  const onClick = useCallback(async () => {
    setDisabled(true);
    const response = await axios.post(API_URL + "/games");
    console.log(response);
    const { game_id: gameId } = response.data;
    console.log(gameId);
    setDisabled(false);
    history.push("/games/" + gameId);
  });

  return (
    <div className="flex flex-col items-center pt-32">
      <h1 className="font-bold text-5xl mb-2">Welcome</h1>
      <button
        disabled={disabled}
        className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
        onClick={onClick}
      >
        Start Game
      </button>
    </div>
  );
}
