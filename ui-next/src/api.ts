import { type Action, type GameState, parseReplay } from "./model";
const base = (
  import.meta.env.CTRON_API_URL ||
  import.meta.env.VITE_API_URL ||
  `${location.protocol}//${location.hostname}:5001`
).replace(/\/$/, "");
async function request(path: string, init?: RequestInit) {
  const response = await fetch(base + "/api" + path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok)
    throw new Error(
      response.status === 404
        ? "Game or move not found. Check the game ID and whether the CLI saved each step."
        : `Server request failed (${response.status}). Check that the API is running.`,
    );
  return response.json();
}
const gamePath = (id: string) => `/games/${encodeURIComponent(id)}`;
export const api = {
  create: (options: object): Promise<{ game_id: string }> =>
    request("/games", { method: "POST", body: JSON.stringify(options) }),
  state: async (
    id: string,
    index: number | "latest" = "latest",
    signal?: AbortSignal,
  ): Promise<GameState> =>
    parseReplay(
      JSON.stringify(
        await request(`${gamePath(id)}/states/${index}`, { signal }),
      ),
    )[0],
  act: (id: string, action?: Action): Promise<GameState> =>
    request(`${gamePath(id)}/actions`, {
      method: "POST",
      body: action ? JSON.stringify(action) : undefined,
    }),
  analyze: (
    id: string,
    index: number,
    signal?: AbortSignal,
  ): Promise<{
    success: boolean;
    probabilities: Record<string, number>;
    error?: string;
  }> => request(`${gamePath(id)}/states/${index}/mcts-analysis`, { signal }),
};
