import { afterEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { Analysis, ReplayControls } from "./Panels";
import { api } from "./api";
import { parseReplay } from "./model";
import snapshots from "../public/example-game.json";
vi.mock("./api", () => ({ api: { analyze: vi.fn() } }));
const frames = parseReplay(JSON.stringify(snapshots));
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
describe("analysis follows the displayed position", () => {
  it("discards stale analysis without automatically running analysis on the next position", async () => {
    let finish!: (r: {
      success: boolean;
      probabilities: Record<string, number>;
    }) => void;
    vi.mocked(api.analyze).mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    const { rerender } = render(<Analysis id="recorded" game={frames[10]} />);
    fireEvent.click(screen.getByRole("button", { name: "Analyze position" }));
    expect(api.analyze).toHaveBeenCalledWith(
      "recorded",
      10,
      expect.any(AbortSignal),
    );
    rerender(<Analysis id="recorded" game={frames[11]} />);
    await act(async () =>
      finish({ success: true, probabilities: { RED: 77 } }),
    );
    expect(screen.queryByText("77%")).toBeNull();
    expect(api.analyze).toHaveBeenCalledTimes(1);
  });
  it("disables server analysis for local imports", () => {
    render(<Analysis game={frames[10]} />);
    expect(
      (
        screen.getByRole("button", {
          name: "Analyze position",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
});

describe("replay arrow shortcuts", () => {
  const props = () => ({
    index: 2,
    max: 4,
    setIndex: vi.fn(),
    playing: true,
    setPlaying: vi.fn(),
    speed: 800,
    setSpeed: vi.fn(),
    busy: false,
  });
  it("steps in both directions, pauses playback, and stops listening on unmount", () => {
    const p = props();
    const { rerender, unmount } = render(<ReplayControls {...p} />);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(p.setIndex).toHaveBeenLastCalledWith(3);
    expect(p.setPlaying).toHaveBeenCalledWith(false);
    rerender(<ReplayControls {...p} index={3} />);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(p.setIndex).toHaveBeenLastCalledWith(2);
    unmount();
    p.setIndex.mockClear();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(p.setIndex).not.toHaveBeenCalled();
  });
  it("stays within the timeline boundaries", () => {
    const p = props();
    const { rerender } = render(<ReplayControls {...p} index={0} />);
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    rerender(<ReplayControls {...p} index={4} />);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(p.setIndex).not.toHaveBeenCalled();
  });
  it("leaves inputs, modified shortcuts, and open dialogs alone", () => {
    const p = props();
    const { rerender } = render(<ReplayControls {...p} />);
    for (const label of ["Go to move", "Replay position", "Replay speed"])
      fireEvent.keyDown(screen.getByLabelText(label), { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowLeft", altKey: true });
    rerender(
      <>
        <ReplayControls {...p} />
        <dialog open>
          <button>Dialog action</button>
        </dialog>
      </>,
    );
    fireEvent.keyDown(screen.getByText("Dialog action"), { key: "ArrowRight" });
    expect(p.setIndex).not.toHaveBeenCalled();
    expect(p.setPlaying).not.toHaveBeenCalled();
  });
});
