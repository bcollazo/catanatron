import { HUMAN_COLOR } from "../constants";

export function isPlayersTurn(gameState) {
  return gameState.current_color === HUMAN_COLOR;
}

export function isInitialPhase(gameState) {
  const buildInitialSettlementActions = gameState.current_playable_actions.filter(
    (action) =>
      action[1] === "BUILD_FIRST_SETTLEMENT" ||
      action[1] === "BUILD_SECOND_SETTLEMENT"
  );
  return buildInitialSettlementActions.length > 0;
}
