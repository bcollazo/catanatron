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

export async function getMctsAnalysis(gameId, stateIndex = 'latest') {
  try {
    console.log('Getting MCTS analysis for:', {
      gameId,
      stateIndex,
      url: `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`
    });

    if (!gameId) {
      throw new Error('No gameId provided to getMctsAnalysis');
    }

    const response = await axios.get(
      `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`
    );
    
    console.log('MCTS analysis response:', response.data);
    return response.data;
  } catch (error) {
    console.error('MCTS analysis error:', {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
      stack: error.stack
    });
    throw error;
  }
}
