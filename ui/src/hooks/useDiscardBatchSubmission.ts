import { useCallback, useState } from "react";
import type {
  DiscardGameAction,
  GameAction,
  GameState,
  ResourceCard,
} from "../utils/api.types";
import { postAction } from "../utils/apiClient";

type SubmitDiscardBatchParams = {
  discardActionType: DiscardGameAction[1];
  gameId: string;
  humanColor: GameAction[0];
  resources: ResourceCard[];
};

export function useDiscardBatchSubmission() {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const submitDiscardBatch = useCallback(
    async ({
      discardActionType,
      gameId,
      humanColor,
      resources,
    }: SubmitDiscardBatchParams): Promise<GameState> => {
      setIsSubmitting(true);
      try {
        let nextGameState: GameState | null = null;
        for (const resource of resources) {
          const action: GameAction = [humanColor, discardActionType, resource];
          nextGameState = await postAction(gameId, action);
        }

        if (nextGameState === null) {
          throw new Error("Discard batch submitted with no resources selected.");
        }

        return nextGameState;
      } finally {
        setIsSubmitting(false);
      }
    },
    [],
  );

  return { isSubmitting, submitDiscardBatch };
}
