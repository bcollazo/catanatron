import { useContext } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { StateProvider, store } from "../store";
import { makeGameState } from "../test/fixtures";
import { getState } from "../utils/apiClient";
import ReplayScreen from "./ReplayScreen";

vi.mock("../utils/apiClient", () => ({
  getState: vi.fn(),
}));
vi.mock("./ZoomableBoard", () => ({
  default: () => <div data-testid="board">Board</div>,
}));
vi.mock("../components/LeftDrawer", () => ({
  default: () => <div>Players</div>,
}));
vi.mock("../components/RightDrawer", () => ({
  default: ({ children }: { children: React.ReactNode }) => (
    <aside>{children}</aside>
  ),
}));
vi.mock("../components/AnalysisBox", () => ({
  default: () => <div>Analysis</div>,
}));

function StateIndexProbe() {
  const { state } = useContext(store);
  return <output data-testid="state-index">{state.gameState?.state_index}</output>;
}

function renderReplayScreen() {
  return render(
    <StateProvider>
      <MemoryRouter initialEntries={["/replays/game-123"]}>
        <Routes>
          <Route path="/replays/:gameId" element={<ReplayScreen />} />
        </Routes>
      </MemoryRouter>
      <StateIndexProbe />
    </StateProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getState).mockImplementation(async (_gameId, stateIndex) => {
    const index = stateIndex === "latest" ? 12 : Number(stateIndex);
    return makeGameState({ state_index: index });
  });
});

afterEach(() => {
  cleanup();
});

test("loads the latest replay boundary and navigates persisted states", async () => {
  const user = userEvent.setup();
  renderReplayScreen();

  expect(await screen.findByTestId("board")).toBeInTheDocument();
  await waitFor(() => {
    expect(screen.getByText("Move: 0 / 12")).toBeInTheDocument();
  });
  expect(getState).toHaveBeenCalledWith("game-123", "latest");
  expect(getState).toHaveBeenCalledWith("game-123", 0);
  expect(screen.getByTestId("state-index")).toHaveTextContent("0");

  await user.click(screen.getByRole("button", { name: "Next Move" }));

  await waitFor(() => {
    expect(getState).toHaveBeenCalledWith("game-123", 1);
    expect(screen.getByTestId("state-index")).toHaveTextContent("1");
    expect(screen.getByText("Move: 1 / 12")).toBeInTheDocument();
  });
});
