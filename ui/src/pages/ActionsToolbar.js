import React, {
  useState,
  useRef,
  useEffect,
  useContext,
  useCallback,
} from "react";
import memoize from "fast-memoize";
import { Button, Hidden } from "@material-ui/core";
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
import Prompt, { humanizeTradeAction } from "../components/Prompt";
import ResourceSelector from "../components/ResourceSelector";
import { store } from "../store";
import ACTIONS from "../actions";
import { getHumanColor, playerKey } from "../utils/stateUtils";
import { postAction } from "../utils/apiClient";

import "./ActionsToolbar.scss";
import { useSnackbar } from "notistack";
import { dispatchSnackbar } from "../components/Snackbar";

function PlayButtons() {
  const { gameId } = useParams();
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const [resourceSelectorOpen, setResourceSelectorOpen] = useState(false);

  const carryOutAction = useCallback(
    memoize((action) => async () => {
      const gameState = await postAction(gameId, action);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
    }),
    [enqueueSnackbar, closeSnackbar]
  );

  const { gameState, isPlayingMonopoly, isPlayingYearOfPlenty } = state;
  const key = playerKey(gameState, gameState.current_color);
  const isRoll =
    gameState.current_prompt === "PLAY_TURN" &&
    !gameState.player_state[`${key}_HAS_ROLLED`];
  const isDiscard = gameState.current_prompt === "DISCARD";
  const isMoveRobber = gameState.current_prompt === "MOVE_ROBBER";
  const playableDevCardTypes = new Set(
    gameState.current_playable_actions
      .filter((action) => action[1].startsWith("PLAY"))
      .map((action) => action[1])
  );
  const humanColor = getHumanColor(state.gameState);
  const setIsPlayingMonopoly = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_PLAYING_MONOPOLY });
  }, [dispatch]);
  const getValidYearOfPlentyOptions = useCallback(() => {
    return state.gameState.current_playable_actions
      .filter(action => action[1] === "PLAY_YEAR_OF_PLENTY")
      .map(action => action[2]);
  }, [state.gameState.current_playable_actions]);  
  const handleResourceSelection = useCallback(async (selectedResources) => {
    setResourceSelectorOpen(false);
    let action;
    if (isPlayingMonopoly) {
      action = [humanColor, "PLAY_MONOPOLY", selectedResources]
    } else if (isPlayingYearOfPlenty) {
      action = [humanColor, "PLAY_YEAR_OF_PLENTY", selectedResources];
    } else {
      console.error('Invalid resource selector mode');
      return;
    }
    const gameState = await postAction(gameId, action);
    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
  }, [gameId, humanColor, dispatch, enqueueSnackbar, closeSnackbar, isPlayingMonopoly, isPlayingYearOfPlenty]);
  const handleOpenResourceSelector = useCallback(() => {
    setResourceSelectorOpen(true);
  }, []);
  const setIsPlayingYearOfPlenty = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_PLAYING_YEAR_OF_PLENTY });
  }, [dispatch]);
  const playRoadBuilding = useCallback(async () => {
    const action = [humanColor, "PLAY_ROAD_BUILDING", null];
    const gameState = await postAction(gameId, action);
    dispatch({ type: ACTIONS.PLAY_ROAD_BUILDING });
    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
  }, [gameId, dispatch, enqueueSnackbar, closeSnackbar, humanColor]);
  const playKnightCard = useCallback(async () => {
    const action = [humanColor, "PLAY_KNIGHT_CARD", null];
    const gameState = await postAction(gameId, action);
    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
  }, [gameId, dispatch, enqueueSnackbar, closeSnackbar, humanColor]);
  const useItems = [
    {
      label: "Monopoly",
      disabled: !playableDevCardTypes.has("PLAY_MONOPOLY"),
      onClick: setIsPlayingMonopoly,
    },
    {
      label: "Year of Plenty",
      disabled: !playableDevCardTypes.has("PLAY_YEAR_OF_PLENTY"),
      onClick: setIsPlayingYearOfPlenty,
    },
    {
      label: "Road Building",
      disabled: !playableDevCardTypes.has("PLAY_ROAD_BUILDING"),
      onClick: playRoadBuilding,
    },
    {
      label: "Knight",
      disabled: !playableDevCardTypes.has("PLAY_KNIGHT_CARD"),
      onClick: playKnightCard,
    },
  ];

  const buildActionTypes = new Set(
    state.gameState.is_initial_build_phase
      ? []
      : state.gameState.current_playable_actions
          .filter(
            (action) =>
              action[1].startsWith("BUY") || action[1].startsWith("BUILD")
          )
          .map((a) => a[1])
  );
  const buyDevCard = useCallback(async () => {
    const action = [humanColor, "BUY_DEVELOPMENT_CARD", null];
    const gameState = await postAction(gameId, action);
    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
  }, [gameId, dispatch, enqueueSnackbar, closeSnackbar, humanColor]);
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
  const tradeItems = React.useMemo(() => {
    const items = tradeActions.map((action) => {
      const label = humanizeTradeAction(action);
      return {
        label: label,
        disabled: false,
        onClick: carryOutAction(action),
      };
    });
  
    return items.sort((a, b) => a.label.localeCompare(b.label));
  }, [tradeActions, carryOutAction]);

  const setIsMovingRobber = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_MOVING_ROBBER });
  }, [dispatch]);
  const rollAction = carryOutAction([humanColor, "ROLL", null]);
  const proceedAction = carryOutAction();
  const endTurnAction = carryOutAction([humanColor, "END_TURN", null]);
  return (
    <>
      <OptionsButton
        disabled={playableDevCardTypes.size === 0 || isPlayingYearOfPlenty}
        menuListId="use-menu-list"
        icon={<SimCardIcon />}
        items={useItems}
      >
        Use
      </OptionsButton>
      <OptionsButton
        disabled={buildActionTypes.size === 0 || isPlayingYearOfPlenty}
        menuListId="build-menu-list"
        icon={<BuildIcon />}
        items={buildItems}
      >
        Buy
      </OptionsButton>
      <OptionsButton
        disabled={tradeItems.length === 0 || isPlayingYearOfPlenty}
        menuListId="trade-menu-list"
        icon={<AccountBalanceIcon />}
        items={tradeItems}
      >
        Trade
      </OptionsButton>
      <Button
        disabled={gameState.is_initial_build_phase}
        variant="contained"
        color="primary"
        startIcon={<NavigateNextIcon />}
        onClick={
          isRoll
            ? rollAction
            : isDiscard
            ? proceedAction
            : isMoveRobber
            ? setIsMovingRobber
            : isPlayingYearOfPlenty || isPlayingMonopoly
            ? handleOpenResourceSelector
            : endTurnAction
        }
      >
        {
          isRoll ? "ROLL" : 
          isDiscard ? "DISCARD" : 
          isMoveRobber ? "ROB" : 
          isPlayingYearOfPlenty || isPlayingMonopoly ? "SELECT" : 
          "END"
        }
      </Button>
      <ResourceSelector
        open={resourceSelectorOpen}
        onClose={() => {
          setResourceSelectorOpen(false);
          dispatch({ type: ACTIONS.CANCEL_MONOPOLY });
          dispatch({ type: ACTIONS.CANCEL_YEAR_OF_PLENTY });
        }}
        options={getValidYearOfPlentyOptions()}
        onSelect={handleResourceSelection}
        mode={isPlayingMonopoly ? 'monopoly' : 'yearOfPlenty'}
      />
    </>
  );
}

export default function ActionsToolbar({
  zoomIn,
  zoomOut,
  isBotThinking,
  replayMode,
}) {
  const { state, dispatch } = useContext(store);

  const openLeftDrawer = useCallback(() => {
    dispatch({
      type: ACTIONS.SET_LEFT_DRAWER_OPENED,
      data: true,
    });
  }, [dispatch]);

  const botsTurn = state.gameState.bot_colors.includes(
    state.gameState.current_color
  );
  const humanColor = getHumanColor(state.gameState);
  return (
    <>
      <div className="state-summary">
        <Hidden mdUp>
          <Button className="open-drawer-btn" onClick={openLeftDrawer}>
            <ChevronLeftIcon />
          </Button>
        </Hidden>
        {humanColor && (
          <ResourceCards
            playerState={state.gameState.player_state}
            playerKey={playerKey(state.gameState, humanColor)}
          />
        )}
      </div>
      <div className="actions-toolbar">
        {!(botsTurn || state.gameState.winning_color) && !replayMode && (
          <PlayButtons gameState={state.gameState} />
        )}
        {(botsTurn || state.gameState.winning_color) && (
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
