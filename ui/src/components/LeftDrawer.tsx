import React, {
  PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import cn from "classnames";
import SwipeableDrawer from "@mui/material/SwipeableDrawer";
import Divider from "@mui/material/Divider";
import Drawer from "@mui/material/Drawer";

import Hidden from "./Hidden";
import PlayerStateBox from "./PlayerStateBox";
import { humanizeActionRecord } from "../utils/promptUtils";
import { store } from "../store";
import ACTIONS from "../actions";
import { playerKey } from "../utils/stateUtils";
import { type Color, type GameState } from "../utils/api.types";
import { isTabOrShift, type InteractionEvent } from "../utils/events";

import "./LeftDrawer.scss";

const LEFT_DRAWER_LS_KEY = "catanatron_left_drawer_width_px";
const LEFT_DRAWER_MIN = 260;
const LEFT_DRAWER_MAX = 720;
const LEFT_DRAWER_DEFAULT = 340;

function readStoredDrawerWidth(): number {
  try {
    const raw = localStorage.getItem(LEFT_DRAWER_LS_KEY);
    if (!raw) return LEFT_DRAWER_DEFAULT;
    const n = parseInt(raw, 10);
    if (!Number.isFinite(n)) return LEFT_DRAWER_DEFAULT;
    return Math.min(Math.max(n, LEFT_DRAWER_MIN), LEFT_DRAWER_MAX);
  } catch {
    return LEFT_DRAWER_DEFAULT;
  }
}

function DrawerContent({
  gameState,
  children,
}: PropsWithChildren<{ gameState: GameState }>) {
  const playerLabel = (color: Color) => {
    const botLabel = gameState.bot_labels?.[color];
    if (botLabel) return botLabel;
    return gameState.bot_colors.includes(color) ? "BOT" : "YOU";
  };

  const playerSections = gameState.colors.map((color) => {
    const key = playerKey(gameState, color);
    return (
      <React.Fragment key={color}>
        <div className={cn("player-label foreground", color)}>
          {playerLabel(color)}
        </div>
        <PlayerStateBox
          playerState={gameState.player_state}
          playerKey={key}
          color={color}
        />
        <Divider />
      </React.Fragment>
    );
  });

  return (
    <>
      {playerSections}
      {children}
      <div className="log">
        {gameState.action_records
          .slice()
          .reverse()
          .map((actionRecord, i) => (
            <div
              key={i}
              className={cn("action foreground", actionRecord[0][0])}
            >
              {humanizeActionRecord(gameState, actionRecord)}
            </div>
          ))}
      </div>
    </>
  );
}

export default function LeftDrawer({ children }: PropsWithChildren) {
  const { state, dispatch } = useContext(store);
  const iOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

  const [paperWidth, setPaperWidth] = useState(readStoredDrawerWidth);
  const dragStartRef = useRef<{ x: number; width: number } | null>(null);

  useEffect(() => {
    document.documentElement.style.setProperty("--left-drawer-width", `${paperWidth}px`);
  }, [paperWidth]);

  const clampWidth = useCallback((w: number) => {
    const max =
      typeof window !== "undefined"
        ? Math.min(LEFT_DRAWER_MAX, Math.floor(window.innerWidth * 0.85))
        : LEFT_DRAWER_MAX;
    return Math.min(Math.max(Math.round(w), LEFT_DRAWER_MIN), max);
  }, []);

  const startResize = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      dragStartRef.current = { x: event.clientX, width: paperWidth };

      const onMove = (ev: MouseEvent) => {
        const start = dragStartRef.current;
        if (!start) return;
        const dx = ev.clientX - start.x;
        setPaperWidth(clampWidth(start.width + dx));
      };

      const onUp = () => {
        dragStartRef.current = null;
        document.body.style.removeProperty("user-select");
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        setPaperWidth((w) => {
          const next = clampWidth(w);
          try {
            localStorage.setItem(LEFT_DRAWER_LS_KEY, String(next));
          } catch {
            /* ignore quota */
          }
          return next;
        });
      };

      document.body.style.setProperty("user-select", "none");
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [clampWidth, paperWidth]
  );

  // Do not set position on Paper — MUI permanent drawers use fixed positioning.
  // Forcing `relative` pulls the panel into normal flow and the docked root becomes
  // a full-width strip (black bar) covering the board area.
  const drawerPaperSx = {
    width: paperWidth,
    boxSizing: "border-box" as const,
    maxWidth: "min(720px, 85vw)",
  };

  const openLeftDrawer = useCallback(
    (event: InteractionEvent) => {
      if (isTabOrShift(event)) {
        return;
      }

      dispatch({ type: ACTIONS.SET_LEFT_DRAWER_OPENED, data: true });
    },
    [dispatch]
  );
  const closeLeftDrawer = useCallback(
    (event: InteractionEvent) => {
      if (isTabOrShift(event)) {
        return;
      }

      dispatch({ type: ACTIONS.SET_LEFT_DRAWER_OPENED, data: false });
    },
    [dispatch]
  );

  const drawerInner = (
    <>
      {/* Right edge = inner edge toward the board; drag horizontally to widen/narrow */}
      <div
        className="left-drawer-resize-handle"
        onMouseDown={startResize}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize left panel"
        aria-valuemin={LEFT_DRAWER_MIN}
        aria-valuemax={LEFT_DRAWER_MAX}
        aria-valuenow={paperWidth}
      />
      <DrawerContent gameState={state.gameState as GameState}>
        {children}
      </DrawerContent>
    </>
  );

  return (
    <>
      <Hidden breakpoint={{ size: "md", direction: "up" }} implementation="js">
        <SwipeableDrawer
          className="left-drawer"
          anchor="left"
          open={state.isLeftDrawerOpen}
          onClose={closeLeftDrawer}
          onOpen={openLeftDrawer}
          disableBackdropTransition={!iOS}
          disableDiscovery={iOS}
          onKeyDown={closeLeftDrawer}
          slotProps={{ paper: { sx: drawerPaperSx } }}
        >
          {drawerInner}
        </SwipeableDrawer>
      </Hidden>
      <Hidden
        breakpoint={{ size: "sm", direction: "down" }}
        implementation="css"
      >
        <Drawer
          className="left-drawer"
          anchor="left"
          variant="permanent"
          open
          slotProps={{ paper: { sx: drawerPaperSx } }}
        >
          {drawerInner}
        </Drawer>
      </Hidden>
    </>
  );
}
