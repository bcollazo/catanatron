import { IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import type { GameState } from "../utils/api.types";
import { humanizeAction } from "../utils/promptUtils";

// No types exported from notistack;
type SnackbarKey = string | number;

export const snackbarActions = (closeSnackbar: (key?: SnackbarKey) => void) => (key: string) =>
  (
    <>
      <IconButton
        size="small"
        aria-label="close"
        color="inherit"
        onClick={() => closeSnackbar(key)}
      >
        <CloseIcon fontSize="small" />
      </IconButton>
    </>
  );

export function dispatchSnackbar(enqueueSnackbar: (message: string, options: { action: (key: string) => React.ReactNode;  onClick: () => void }) => SnackbarKey , closeSnackbar: (key?: string | number) => void, gameState: GameState) {
  enqueueSnackbar(humanizeAction(gameState, gameState.actions.slice(-1)[0]), {
    action: snackbarActions(closeSnackbar),
    onClick: () => {
      closeSnackbar();
    },
  });
}
