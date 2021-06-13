import axios from "axios";

import { API_URL } from "../configuration";

export async function createGame(players) {
  const response = await axios.post(API_URL + "/api/games", { players });
  return response.data.game_id;
}

export async function getState(gameId, stateIndex = "latest") {
  const response = await axios.get(
    `${API_URL}/api/games/${gameId}/states/${stateIndex}`
  );
  return response.data;
}

/** action=undefined means bot action */
export async function postAction(gameId, action = undefined) {
  const response = await axios.post(
    `${API_URL}/api/games/${gameId}/actions`,
    action
  );
  return response.data;
}
