import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import snapshots from "../public/example-game.json";
import { parseReplay, type GameState } from "./model";
import { api } from "./api";
import { useGame, type Session } from "./useGame";
vi.mock("./api", () => ({ api: { state: vi.fn(), act: vi.fn() } }));
const frames = parseReplay(JSON.stringify(snapshots));
const human = { ...frames[0], bot_colors: [] } as GameState;
const deferred = <T,>() => {
  let resolve!: (v: T) => void;
  let reject!: (e: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.state).mockResolvedValue(human);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});
describe("game state coordination", () => {
  it("sends only one mutation for a rapid double click", async () => {
    const session: Session = { id: "test", replay: false };
    const request = deferred<GameState>();
    vi.mocked(api.act).mockReturnValue(request.promise);
    const { result } = renderHook(() => useGame(session));
    await waitFor(() => expect(result.current.game).toBe(human));
    act(() => {
      void result.current.act(human.current_playable_actions[0]);
      void result.current.act(human.current_playable_actions[0]);
    });
    expect(api.act).toHaveBeenCalledTimes(1);
    await act(async () => request.resolve({ ...frames[1], bot_colors: [] }));
    expect(result.current.game?.state_index).toBe(1);
  });
  it("rejects moves that are not in the authoritative legal-action list", async () => {
    const session: Session = { id: "test", replay: false };
    const { result } = renderHook(() => useGame(session));
    await waitFor(() => expect(result.current.game).toBe(human));
    await act(async () => result.current.act(["RED", "BUILD_CITY", 999]));
    expect(api.act).not.toHaveBeenCalled();
  });
  it("never advances bots or sends actions from a local replay", async () => {
    vi.useFakeTimers();
    const session: Session = { local: frames, replay: true };
    const { result } = renderHook(() => useGame(session));
    await act(async () => {
      await result.current.act(frames[0].current_playable_actions[0]);
      vi.advanceTimersByTime(5000);
    });
    expect(api.act).not.toHaveBeenCalled();
    expect(api.state).not.toHaveBeenCalled();
    act(() => result.current.setIndex(20));
    expect(result.current.game?.state_index).toBe(20);
  });
  it("ignores a slow old replay response after seeking to a newer move", async () => {
    const old = deferred<GameState>(),
      next = deferred<GameState>();
    vi.mocked(api.state).mockImplementation((_, index) =>
      index === "latest"
        ? Promise.resolve(frames[80])
        : index === 0
          ? old.promise
          : next.promise,
    );
    const session: Session = { id: "test", replay: true };
    const { result } = renderHook(() => useGame(session));
    await waitFor(() => expect(result.current.max).toBe(80));
    act(() => result.current.setIndex(12));
    await act(async () => next.resolve(frames[12]));
    await act(async () => old.resolve(frames[0]));
    expect(result.current.game?.state_index).toBe(12);
  });
  it("stops automatic retries after an uncertain mutation failure", async () => {
    vi.mocked(api.state).mockResolvedValue(frames[0]);
    vi.mocked(api.act).mockRejectedValue(new Error("Connection lost"));
    const session: Session = { id: "test", replay: false };
    const { result } = renderHook(() => useGame(session));
    await waitFor(() => expect(result.current.game).toBe(frames[0]));
    await act(async () => result.current.act());
    expect(result.current.paused).toBe(true);
    expect(result.current.error).toContain("server may have received");
    expect(api.act).toHaveBeenCalledTimes(1);
  });
});
