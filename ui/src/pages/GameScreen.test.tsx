import { useContext } from "react";
import { act, cleanup, render, screen } from "@testing-library/react";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { StateProvider, store } from "../store";
import type { GameState } from "../utils/api.types";
import { getState, postAction, setPerspective } from "../utils/apiClient";
import { dispatchSnackbar } from "../components/Snackbar";
import { makeGameState } from "../test/fixtures";
import GameScreen from "./GameScreen";

const snackbar = vi.hoisted(() => ({
  enqueueSnackbar: vi.fn(),
  closeSnackbar: vi.fn(),
}));

vi.mock("../utils/apiClient", () => ({
  getState: vi.fn(),
  postAction: vi.fn(),
  getMctsAnalysis: vi.fn(),
  setPerspective: vi.fn(),
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

const theme = createTheme();
const baseState = makeGameState();

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function StateIndexProbe() {
  const { state } = useContext(store);
  return <output data-testid="state-index">{state.gameState?.state_index}</output>;
}

function renderGameScreen({
  initialEntry = "/games/game-123",
  routePath = "/games/:gameId",
  replayMode = false,
}: {
  initialEntry?: string;
  routePath?: string;
  replayMode?: boolean;
} = {}) {
  return render(
    <ThemeProvider theme={theme}>
      <StateProvider>
        <MemoryRouter initialEntries={[initialEntry]}>
          <Routes>
            <Route
              path={routePath}
              element={<GameScreen replayMode={replayMode} />}
            />
          </Routes>
        </MemoryRouter>
        <StateIndexProbe />
      </StateProvider>
    </ThemeProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

test("loads a persisted API state into the real gameplay controls", async () => {
  vi.mocked(getState).mockResolvedValue(baseState);

  renderGameScreen();

  expect(await screen.findByTestId("board")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "ROLL" })).toBeEnabled();
  expect(screen.getAllByTitle("Victory Points").length).toBeGreaterThan(0);
  expect(screen.getByTestId("state-index")).toHaveTextContent("12");
  expect(getState).toHaveBeenCalledWith("game-123", undefined);
  // Spectator by default: every request carries the perspective, so it is set
  // before the first fetch rather than passed at each call site.
  expect(setPerspective).toHaveBeenCalledWith(null);
  expect(postAction).not.toHaveBeenCalled();
});

test("advances a persisted bot turn and publishes the returned state", async () => {
  const botTurn = makeGameState({ current_color: "RED" });
  const humanTurn: GameState = makeGameState({
    state_index: 13,
    action_records: [
      [["RED", "END_TURN", null], null],
    ] as GameState["action_records"],
  });
  const actionResponse = deferred<GameState>();
  vi.mocked(getState).mockResolvedValue(botTurn);
  vi.mocked(postAction).mockReturnValue(actionResponse.promise);

  renderGameScreen();

  expect(await screen.findByTestId("board")).toBeInTheDocument();
  expect(postAction).toHaveBeenCalledWith("game-123");

  vi.useFakeTimers();
  await act(async () => {
    actionResponse.resolve(humanTurn);
    await actionResponse.promise;
  });

  await act(async () => {
    await vi.runAllTimersAsync();
  });

  expect(screen.getByTestId("state-index")).toHaveTextContent("13");
  expect(screen.getByRole("button", { name: "ROLL" })).toBeEnabled();
  expect(dispatchSnackbar).toHaveBeenCalledOnce();
});

test("loads a historical state without advancing bots in replay mode", async () => {
  vi.mocked(getState).mockResolvedValue(makeGameState({ state_index: 7 }));

  renderGameScreen({
    initialEntry: "/games/game-123/states/7",
    routePath: "/games/:gameId/states/:stateIndex",
    replayMode: true,
  });

  expect(await screen.findByTestId("board")).toBeInTheDocument();
  expect(screen.getByTestId("state-index")).toHaveTextContent("7");
  expect(getState).toHaveBeenCalledWith("game-123", "7");
  expect(postAction).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: "ROLL" })).not.toBeInTheDocument();
});
