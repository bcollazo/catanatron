import React, {
  useState,
  useRef,
  useEffect,
  useContext,
  useCallback,
} from "react";
import memoize from "fast-memoize";
import { Button } from "@mui/material";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import AccountBalanceIcon from "@mui/icons-material/AccountBalance";
import BuildIcon from "@mui/icons-material/Build";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import MenuItem from "@mui/material/MenuItem";
import ClickAwayListener from "@mui/material/ClickAwayListener";
import Grow from "@mui/material/Grow";
import Paper from "@mui/material/Paper";
import Popper from "@mui/material/Popper";
import MenuList from "@mui/material/MenuList";
import SimCardIcon from "@mui/icons-material/SimCard";
import { useParams } from "react-router";

import Hidden from "../components/Hidden";
import Prompt from "../components/Prompt";
import ResourceCards from "../components/ResourceCards";
import ResourceSelector from "../components/ResourceSelector";
import { store } from "../store";
import ACTIONS from "../actions";
import type { GameAction, ResourceCard } from "../utils/api.types"; // Add GameState to the import, adjust path if needed
import { getHumanColor, playerKey } from "../utils/stateUtils";
import { postAction } from "../utils/apiClient";
import { humanizeTradeAction } from "../utils/promptUtils";

import "./ActionsToolbar.scss";
import { useSnackbar } from "notistack";
import { dispatchSnackbar } from "../components/Snackbar";

function PlayButtons() {
  const { gameId } = useParams();
  if (!gameId) {
    console.error("Game ID is not found in URL parameters.");
    return null;
  }
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const [resourceSelectorOpen, setResourceSelectorOpen] = useState(false);

  const carryOutAction = useCallback(
    memoize((action?: GameAction) => async () => {
      const gameState = await postAction(gameId, action);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
    }),
    [enqueueSnackbar, closeSnackbar]
  );

  const {
    gameState,
    isPlayingMonopoly,
    isPlayingYearOfPlenty,
    isRoadBuilding,
  } = state;
  if (gameState === null) {
    return null;
  }
  const key = playerKey(gameState, gameState.current_color);
  const isRoll =
    gameState.current_prompt === "PLAY_TURN" &&
    !gameState.player_state[`${key}_HAS_ROLLED`];
  const isDiscard = gameState.current_prompt === "DISCARD";
  const isMoveRobber = gameState.current_prompt === "MOVE_ROBBER";
  const isPlayingDevCard =
    isPlayingMonopoly || isPlayingYearOfPlenty || isRoadBuilding;
  const playableDevCardTypes = new Set(
    gameState.current_playable_actions
      .filter((action) => action[1].startsWith("PLAY"))
      .map((action) => action[1])
  );
  const humanColor = getHumanColor(gameState);
  const setIsPlayingMonopoly = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_PLAYING_MONOPOLY });
  }, [dispatch]);
  const getValidYearOfPlentyOptions = useCallback(() => {
    return gameState.current_playable_actions
      .filter((action) => action[1] === "PLAY_YEAR_OF_PLENTY")
      .map((action) => action[2]);
  }, [gameState.current_playable_actions]);
  const handleResourceSelection = useCallback(
    async (selectedResources: ResourceCard | ResourceCard[]) => {
      setResourceSelectorOpen(false);
      let action: GameAction;
      if (isPlayingMonopoly) {
        action = [
          humanColor,
          "PLAY_MONOPOLY",
          selectedResources as ResourceCard,
        ];
      } else if (isPlayingYearOfPlenty) {
        action = [
          humanColor,
          "PLAY_YEAR_OF_PLENTY",
          selectedResources as [ResourceCard] | [ResourceCard, ResourceCard],
        ];
      } else {
        console.error("Invalid resource selector mode");
        return;
      }
      const gameState = await postAction(gameId, action);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
      dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
    },
    [
      gameId,
      humanColor,
      dispatch,
      enqueueSnackbar,
      closeSnackbar,
      isPlayingMonopoly,
      isPlayingYearOfPlenty,
    ]
  );
  const handleOpenResourceSelector = useCallback(() => {
    setResourceSelectorOpen(true);
  }, []);
  const setIsPlayingYearOfPlenty = useCallback(() => {
    dispatch({ type: ACTIONS.SET_IS_PLAYING_YEAR_OF_PLENTY });
  }, [dispatch]);
  const playRoadBuilding = useCallback(async () => {
    const action: GameAction = [humanColor, "PLAY_ROAD_BUILDING", null];
    const gameState = await postAction(gameId, action);
    dispatch({ type: ACTIONS.PLAY_ROAD_BUILDING });
    dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
  }, [gameId, dispatch, enqueueSnackbar, closeSnackbar, humanColor]);
  const playKnightCard = useCallback(async () => {
    const action: GameAction = [humanColor, "PLAY_KNIGHT_CARD", null];
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
    gameState.is_initial_build_phase
      ? []
      : gameState.current_playable_actions
          .filter(
            (action) =>
              action[1].startsWith("BUY") || action[1].startsWith("BUILD")
          )
          .map((a) => a[1])
  );
  const buyDevCard = useCallback(async () => {
    const action: GameAction = [humanColor, "BUY_DEVELOPMENT_CARD", null];
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
  const toggleBuildingRoad = useCallback(() => {
    dispatch({ type: ACTIONS.TOGGLE_BUILDING_ROAD });
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
      onClick: toggleBuildingRoad,
    },
  ];

  const tradeActions = gameState.current_playable_actions.filter(
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
        disabled={playableDevCardTypes.size === 0 || isPlayingDevCard}
        menuListId="use-menu-list"
        icon={<SimCardIcon />}
        items={useItems}
      >
        Use
      </OptionsButton>
      <OptionsButton
        disabled={buildActionTypes.size === 0 || isPlayingDevCard}
        menuListId="build-menu-list"
        icon={<BuildIcon />}
        items={buildItems}
      >
        Buy
      </OptionsButton>
      <OptionsButton
        disabled={tradeItems.length === 0 || isPlayingDevCard}
        menuListId="trade-menu-list"
        icon={<AccountBalanceIcon />}
        items={tradeItems}
      >
        Trade
      </OptionsButton>
      <Button
        disabled={gameState.is_initial_build_phase || isRoadBuilding}
        variant="contained"
        color="primary"
        startIcon={<NavigateNextIcon />}
        onClick={
          isDiscard
            ? proceedAction
            : isMoveRobber
            ? setIsMovingRobber
            : isPlayingYearOfPlenty || isPlayingMonopoly
            ? handleOpenResourceSelector
            : isRoll
            ? rollAction
            : endTurnAction
        }
      >
        {isDiscard
          ? "DISCARD"
          : isMoveRobber
          ? "ROB"
          : isPlayingYearOfPlenty || isPlayingMonopoly
          ? "SELECT"
          : isRoll
          ? "ROLL"
          : "END"}
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
        mode={isPlayingMonopoly ? "monopoly" : "yearOfPlenty"}
      />
    </>
  );
}

export default function ActionsToolbar({
  isBotThinking,
  replayMode,
}: {
  isBotThinking: boolean;
  replayMode: boolean;
}) {
  const { state, dispatch } = useContext(store);
  const { gameState } = state;
  if (gameState === null) {
    console.error("No gameState found...");
    return null;
  }
  const openLeftDrawer = useCallback(() => {
    dispatch({
      type: ACTIONS.SET_LEFT_DRAWER_OPENED,
      data: true,
    });
  }, [dispatch]);

  const openRightDrawer = useCallback(() => {
    dispatch({
      type: ACTIONS.SET_RIGHT_DRAWER_OPENED,
      data: true,
    });
  }, [dispatch]);

  const botsTurn = gameState.bot_colors.includes(gameState.current_color);
  const humanColor = getHumanColor(gameState);
  return (
    <>
      <div className="state-summary">
        <Hidden breakpoint={{ size: "md", direction: "up" }}>
          <Button className="open-drawer-btn" onClick={openLeftDrawer}>
            <ChevronLeftIcon />
          </Button>
        </Hidden>
        {humanColor && (
          <ResourceCards
            playerState={gameState.player_state}
            playerKey={playerKey(gameState, humanColor)}
          />
        )}
        <Hidden breakpoint={{ size: "lg", direction: "up" }}>
          <Button
            className="open-drawer-btn"
            onClick={openRightDrawer}
            style={{ marginLeft: "auto" }}
          >
            <ChevronRightIcon />
          </Button>
        </Hidden>
      </div>
      <div className="actions-toolbar">
        {!(botsTurn || gameState.winning_color) && !replayMode && (
          <PlayButtons />
        )}
        {(botsTurn || gameState.winning_color) && (
          <Prompt gameState={gameState} isBotThinking={isBotThinking} />
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

type OptionItem = {
  label: string;
  disabled: boolean;
  onClick: (event: MouseEvent | TouchEvent) => void;
};

type OptionsButtonProps = {
  menuListId: string;
  icon: any;
  children: React.ReactNode;
  items: OptionItem[];
  disabled: boolean;
};

function OptionsButton({
  menuListId,
  icon,
  children,
  items,
  disabled,
}: OptionsButtonProps) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLAnchorElement>(null);

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };
  const handleClose =
    (onClick?: (event: MouseEvent | TouchEvent) => void) =>
    (event: MouseEvent | TouchEvent) => {
      if (
        anchorRef.current &&
        anchorRef.current.contains(event.target as Node)
      ) {
        return;
      }

      onClick && onClick(event);
      setOpen(false);
    };
  function handleListKeyDown(event: React.KeyboardEvent) {
    if (event.key === "Tab") {
      event.preventDefault();
      setOpen(false);
    }
  }
  // return focus to the button when we transitioned from !open -> open
  const prevOpen = useRef(open);
  useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current && anchorRef.current.focus();
    }

    prevOpen.current = open;
  }, [open]);

  return (
    <React.Fragment>
      <Button
        disabled={disabled}
        ref={anchorRef}
        href="#"
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
              <ClickAwayListener onClickAway={handleClose()}>
                <MenuList
                  autoFocusItem={open}
                  id={menuListId}
                  onKeyDown={handleListKeyDown}
                >
                  {items.map((item) => (
                    <MenuItem
                      key={item.label}
                      onClick={
                        handleClose(
                          item.onClick
                        ) as unknown as React.MouseEventHandler
                      }
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
