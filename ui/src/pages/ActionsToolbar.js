import React, {
  useState,
  useRef,
  useEffect,
  useContext,
  useCallback,
} from "react";
import memoize from "fast-memoize";
import { Button } from "@material-ui/core";
import ChevronLeftIcon from "@material-ui/icons/ChevronLeft";
import AccountBalanceIcon from "@material-ui/icons/AccountBalance";
import BuildIcon from "@material-ui/icons/Build";
import NavigateNextIcon from "@material-ui/icons/NavigateNext";
import MenuItem from "@material-ui/core/MenuItem";
import ClickAwayListener from "@material-ui/core/ClickAwayListener";
import Grow from "@material-ui/core/Grow";
import Paper from "@material-ui/core/Paper";
import Popper from "@material-ui/core/Popper";
import MenuList from "@material-ui/core/MenuList";
import SimCardIcon from "@material-ui/icons/SimCard";
import { useParams } from "react-router";

import { ResourceCards } from "../components/PlayerStateBox";
import Prompt, { humanizeAction } from "../components/Prompt";
import { BOT_COLOR, HUMAN_COLOR } from "../constants";
import { store } from "../store";
import ACTIONS from "../actions";
import { isInitialPhase } from "../utils/stateUtils";
import { postAction } from "../utils/apiClient";

import "./ActionsToolbar.scss";
import { useSnackbar } from "notistack";
import { snackbarActions } from "../components/Snackbar";

function PlayButtons() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();

  const carryOutAction = useCallback(
    memoize((action) => async () => {
      const gameState = await postAction(gameId, action);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      enqueueSnackbar(humanizeAction(gameState.actions.slice(-1)[0]), {
        action: snackbarActions(closeSnackbar),
      });
    }),
    [enqueueSnackbar, closeSnackbar]
  );

  const isRoll = state.gameState.current_prompt === "ROLL";
  const isDiscard = state.gameState.current_prompt === "DISCARD";
  const isMove = state.gameState.current_prompt === "MOVE_ROBBER";
  const playableDevCardTypes = new Set(
    state.gameState.current_playable_actions
      .filter((action) => action[1].startsWith("PLAY"))
      .map((action) => action[1])
  );
  const useItems = [
    {
      label: "Monopoly",
      disabled: !playableDevCardTypes.has("PLAY_MONOPOLY"),
    },
    {
      label: "Year of Plenty",
      disabled: !playableDevCardTypes.has("PLAY_YEAR_OF_PLENTY"),
    },
    {
      label: "Road Building",
      disabled: !playableDevCardTypes.has("PLAY_ROAD_BUILDING"),
    },
    {
      label: "Knight",
      disabled: !playableDevCardTypes.has("PLAY_KNIGHT_CARD"),
    },
  ];

  const buildActionTypes = new Set(
    state.gameState.current_playable_actions
      .filter(
        (action) =>
          (action[1].startsWith("BUY") || action[1].startsWith("BUILD")) &&
          !action[1].includes("FIRST") &&
          !action[1].includes("SECOND") &&
          !action[1].includes("INITIAL")
      )
      .map((a) => a[1])
  );
  const buyDevCard = useCallback(async () => {
    const action = [HUMAN_COLOR, "BUY_DEVELOPMENT_CARD", null];
    const gameState = await postAction(gameId, action);
    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    enqueueSnackbar(humanizeAction(gameState.actions.slice(-1)[0]), {
      action: snackbarActions(closeSnackbar),
    });
  }, [gameId, dispatch, enqueueSnackbar, closeSnackbar]);
  const setIsBuildingSettlement = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_BUILDING_SETTLEMENT });
  }, [dispatch]);
  const setIsBuildingCity = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_BUILDING_CITY });
  }, [dispatch]);
  const setIsBuildingRoad = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_BUILDING_ROAD });
  }, [dispatch]);
  const buildItems = [
    {
      label: "Development Card",
      disabled: !buildActionTypes.has("BUY_DEVELOPMENT_CARD"),
      onClick: buyDevCard,
    },
    {
      label: "City",
      disabled: !buildActionTypes.has("BUILD_CITY"),
      onClick: setIsBuildingCity,
    },
    {
      label: "Settlement",
      disabled: !buildActionTypes.has("BUILD_SETTLEMENT"),
      onClick: setIsBuildingSettlement,
    },
    {
      label: "Road",
      disabled: !buildActionTypes.has("BUILD_ROAD"),
      onClick: setIsBuildingRoad,
    },
  ];

  const tradeActions = state.gameState.current_playable_actions.filter(
    (action) => action[1] === "MARITIME_TRADE"
  );
  const tradeItems = tradeActions.map((action) => {
    const out = action[2].slice(0, 4).filter((resource) => resource !== null);
    return {
      label: `${out.length} ${out[0]} => ${action[2][4]}`,
      disabled: false,
      onClick: carryOutAction(action),
    };
  });

  const rollAction = carryOutAction([HUMAN_COLOR, "ROLL", null]);
  const proceedAction = carryOutAction();
  const endTurnAction = carryOutAction([HUMAN_COLOR, "END_TURN", null]);
  return (
    <>
      <OptionsButton
        disabled={playableDevCardTypes.size === 0}
        menuListId="use-menu-list"
        icon={<SimCardIcon />}
        items={useItems}
      >
        Use
      </OptionsButton>
      <OptionsButton
        disabled={buildActionTypes.size === 0}
        menuListId="build-menu-list"
        icon={<BuildIcon />}
        items={buildItems}
      >
        Buy
      </OptionsButton>
      <OptionsButton
        disabled={tradeItems.length === 0}
        menuListId="trade-menu-list"
        icon={<AccountBalanceIcon />}
        items={tradeItems}
      >
        Trade
      </OptionsButton>
      <Button
        disabled={isInitialPhase(state.gameState)}
        variant="contained"
        color="primary"
        startIcon={<NavigateNextIcon />}
        onClick={
          isRoll
            ? rollAction
            : isDiscard || isMove
            ? proceedAction
            : endTurnAction
        }
      >
        {isRoll ? "ROLL" : isDiscard ? "DISCARD" : isMove ? "ROB" : "END"}
      </Button>
    </>
  );
}

export default function ActionsToolbar({ zoomIn, zoomOut, isBotThinking }) {
  const { state, dispatch } = useContext(store);

  const openLeftDrawer = useCallback(() => {
    dispatch({
      type: ACTIONS.SET_LEFT_DRAWER_OPENED,
      data: true,
    });
  }, [dispatch]);

  const botsTurn = state.gameState.current_color === BOT_COLOR;
  const human =
    state.gameState &&
    state.gameState.players.find((x) => x.color === HUMAN_COLOR);
  return (
    <>
      <div className="state-summary">
        <Button className="open-drawer-btn" onClick={openLeftDrawer}>
          <ChevronLeftIcon />
        </Button>
        <ResourceCards playerState={human} />
      </div>
      <div className="actions-toolbar">
        {!botsTurn && <PlayButtons gameState={state.gameState} />}
        {botsTurn && (
          <Prompt gameState={state.gameState} isBotThinking={isBotThinking} />
        )}
        {/* <Button
          disabled={disabled}
          className="confirm-btn"
          variant="contained"
          color="primary"
          onClick={onTick}
        >
          Ok
        </Button> */}

        {/* <Button onClick={zoomIn}>Zoom In</Button>
      <Button onClick={zoomOut}>Zoom Out</Button> */}
      </div>
    </>
  );
}

function OptionsButton({ menuListId, icon, children, items, disabled }) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef(null);

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };
  const handleClose = (onClick) => (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }

    onClick && onClick();
    setOpen(false);
  };
  function handleListKeyDown(event) {
    if (event.key === "Tab") {
      event.preventDefault();
      setOpen(false);
    }
  }
  // return focus to the button when we transitioned from !open -> open
  const prevOpen = useRef(open);
  useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }

    prevOpen.current = open;
  }, [open]);

  return (
    <React.Fragment>
      <Button
        disabled={disabled}
        ref={anchorRef}
        aria-controls={open ? menuListId : undefined}
        aria-haspopup="true"
        variant="contained"
        color="secondary"
        startIcon={icon}
        onClick={handleToggle}
      >
        {children}
      </Button>
      <Popper
        className="action-popover"
        open={open}
        anchorEl={anchorRef.current}
        role={undefined}
        transition
        disablePortal
      >
        {({ TransitionProps, placement }) => (
          <Grow
            {...TransitionProps}
            style={{
              transformOrigin:
                placement === "bottom" ? "center top" : "center bottom",
            }}
          >
            <Paper>
              <ClickAwayListener onClickAway={handleClose}>
                <MenuList
                  autoFocusItem={open}
                  id={menuListId}
                  onKeyDown={handleListKeyDown}
                >
                  {items.map((item) => (
                    <MenuItem
                      key={item.label}
                      onClick={handleClose(item.onClick)}
                      disabled={item.disabled}
                    >
                      {item.label}
                    </MenuItem>
                  ))}
                </MenuList>
              </ClickAwayListener>
            </Paper>
          </Grow>
        )}
      </Popper>
    </React.Fragment>
  );
}
