import React from "react";
import { IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
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
