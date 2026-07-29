import { useContext } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { StateProvider, store } from "../store";
import type { GameState } from "../utils/api.types";
import { getState, postAction } from "../utils/apiClient";
import { dispatchSnackbar } from "../components/Snackbar";
import GameScreen from "./GameScreen";

const snackbar = vi.hoisted(() => ({
  enqueueSnackbar: vi.fn(),
  closeSnackbar: vi.fn(),
}));

vi.mock("../utils/apiClient", () => ({
  getState: vi.fn(),
  postAction: vi.fn(),
}));
vi.mock("../components/Snackbar", () => ({
  dispatchSnackbar: vi.fn(),
}));
vi.mock("notistack", () => ({
  useSnackbar: () => snackbar,
}));
vi.mock("./ZoomableBoard", () => ({
  default: () => <div data-testid="board">Board</div>,
}));
vi.mock("./ActionsToolbar", () => ({
  default: () => <div data-testid="actions-toolbar">Actions</div>,
}));
vi.mock("../components/LeftDrawer", () => ({
  default: () => <div data-testid="left-drawer">Players</div>,
}));
vi.mock("../components/RightDrawer", () => ({
  default: ({ children }: { children: React.ReactNode }) => (
    <aside>{children}</aside>
  ),
}));
vi.mock("../components/AnalysisBox", () => ({
  default: () => <div data-testid="analysis-box">Analysis</div>,
}));

const baseState: GameState = {
  tiles: [],
  adjacent_tiles: {},
  bot_colors: ["RED"],
  colors: ["RED", "BLUE"],
  current_color: "BLUE",
  winning_color: undefined,
  current_prompt: "PLAY_TURN",
  player_state: {},
  action_records: [],
  robber_coordinate: [0, 0, 0],
  current_discard_count: 0,
  nodes: [],
  edges: [],
  current_playable_actions: [],
  is_initial_build_phase: false,
  state_index: 12,
};

function StateIndexProbe() {
  const { state } = useContext(store);
  return <output data-testid="state-index">{state.gameState?.state_index}</output>;
}

function renderGameScreen() {
  return render(
    <StateProvider>
      <MemoryRouter initialEntries={["/games/game-123"]}>
        <Routes>
          <Route
            path="/games/:gameId"
            element={<GameScreen replayMode={false} />}
          />
        </Routes>
      </MemoryRouter>
      <StateIndexProbe />
    </StateProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
});

test("loads a persisted API state and renders the gameplay screen", async () => {
  vi.mocked(getState).mockResolvedValue(baseState);

  renderGameScreen();

  expect(await screen.findByTestId("board")).toBeInTheDocument();
  expect(screen.getByTestId("actions-toolbar")).toBeInTheDocument();
  expect(screen.getByTestId("left-drawer")).toBeInTheDocument();
  expect(screen.getByTestId("analysis-box")).toBeInTheDocument();
  expect(screen.getByTestId("state-index")).toHaveTextContent("12");
  expect(getState).toHaveBeenCalledWith("game-123", undefined);
  expect(postAction).not.toHaveBeenCalled();
});

test("advances a persisted bot turn and publishes the returned state", async () => {
  const botTurn = { ...baseState, current_color: "RED" as const };
  const humanTurn = {
    ...baseState,
    state_index: 13,
    action_records: [
      [["RED", "END_TURN", null], null],
    ] as GameState["action_records"],
  };
  vi.mocked(getState).mockResolvedValue(botTurn);
  vi.mocked(postAction).mockResolvedValue(humanTurn);

  renderGameScreen();

  await waitFor(() => {
    expect(postAction).toHaveBeenCalledWith("game-123");
  });
  await waitFor(
    () => {
      expect(screen.getByTestId("state-index")).toHaveTextContent("13");
    },
    { timeout: 1000 }
  );
  expect(dispatchSnackbar).toHaveBeenCalledOnce();
});
