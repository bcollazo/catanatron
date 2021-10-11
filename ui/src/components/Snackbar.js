import React from "react";
import { IconButton } from "@material-ui/core";
import CloseIcon from "@material-ui/icons/Close";
import { humanizeAction } from "./Prompt";

export const snackbarActions = (closeSnackbar) => (key) =>
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

export function dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState) {
  enqueueSnackbar(humanizeAction(gameState, gameState.actions.slice(-1)[0]), {
    action: snackbarActions(closeSnackbar),
    onClick: () => {
      closeSnackbar();
    },
  });
}
