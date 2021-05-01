import axios from "axios";

import { API_URL } from "../configuration";

export async function createGame() {
  const response = await axios.post(API_URL + "/api/games");
  return response.data.game_id;
}

export async function getState(gameId) {
  const response = await axios.get(API_URL + "/api/games/" + gameId);
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
