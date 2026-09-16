import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { type Action, type GameState, same } from "./model";

export interface Session {
  id?: string;
  local?: GameState[];
  replay: boolean;
  index?: number;
  label?: string;
}
export function useGame(session: Session) {
  const [game, setGame] = useState<GameState | null>(
    session.local?.[0] ?? null,
  );
  const [busy, setBusy] = useState(false);
  const [loadingLatest, setLoadingLatest] = useState(false);
  const [loadingReplay, setLoadingReplay] = useState(false);
  const [error, setError] = useState("");
  const [index, setIndex] = useState(session.index ?? 0);
  const [max, setMax] = useState(session.local ? session.local.length - 1 : 0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(800);
  const [paused, setPaused] = useState(false);
  const lock = useRef(false);
  const uncertain = useRef(false);
  const refreshRequest = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const current = useRef(game);
  current.current = game;
  useEffect(() => () => refreshRequest.current?.abort(), [index]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  const refresh = useCallback(async () => {
    if (!session.id || lock.current) return;
    lock.current = true;
    const controller = new AbortController();
    refreshRequest.current = controller;
    setBusy(true);
    setError("");
    try {
      const latest = await api.state(session.id, "latest", controller.signal);
      const visible = session.replay
        ? await api.state(session.id, index, controller.signal)
        : latest;
      if (mounted.current && !controller.signal.aborted) {
        uncertain.current = false;
        current.current = visible;
        setMax(latest.state_index);
        setGame(visible);
      }
    } catch (e) {
      if (mounted.current && !controller.signal.aborted)
        setError((e as Error).message);
    } finally {
      lock.current = false;
      if (mounted.current) setBusy(false);
    }
  }, [session.id, session.replay, index]);
  useEffect(() => {
    if (!session.id) return;
    const controller = new AbortController();
    setLoadingLatest(true);
    setError("");
    (async () => {
      try {
        const latest = await api.state(
          session.id!,
          "latest",
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setMax(latest.state_index);
        if (!session.replay) setGame(latest);
      } catch (e) {
        if (!controller.signal.aborted) setError((e as Error).message);
      } finally {
        if (!controller.signal.aborted) setLoadingLatest(false);
      }
    })();
    return () => controller.abort();
  }, [session]);
  useEffect(() => {
    if (!session.replay) return;
    if (session.local) {
      setGame(session.local[index]);
      return;
    }
    if (!session.id) return;
    const controller = new AbortController();
    setLoadingReplay(true);
    setError("");
    api
      .state(session.id, index, controller.signal)
      .then((s) => {
        if (!controller.signal.aborted) setGame(s);
      })
      .catch((e) => {
        if (!controller.signal.aborted) {
          setError(e.message);
          setPlaying(false);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingReplay(false);
      });
    return () => controller.abort();
  }, [index, session]);
  const act = useCallback(
    async (action?: Action) => {
      if (
        lock.current ||
        uncertain.current ||
        session.replay ||
        !session.id ||
        !current.current ||
        current.current.winning_color
      )
        return;
      if (
        !action &&
        !current.current.bot_colors.includes(current.current.current_color)
      )
        return;
      if (
        action &&
        !current.current.current_playable_actions.some((a) => same(a, action))
      )
        return;
      lock.current = true;
      setBusy(true);
      setError("");
      try {
        const next = await api.act(session.id, action);
        if (mounted.current) {
          current.current = next;
          setGame(next);
          setMax(next.state_index);
        }
      } catch (e) {
        uncertain.current = true;
        if (mounted.current) {
          setError(
            `${(e as Error).message} Reload the game before trying another move; the server may have received it.`,
          );
          setPaused(true);
        }
      } finally {
        lock.current = false;
        if (mounted.current) setBusy(false);
      }
    },
    [session],
  );
  useEffect(() => {
    if (
      session.replay ||
      !game ||
      busy ||
      paused ||
      error ||
      game.winning_color ||
      !game.bot_colors.includes(game.current_color)
    )
      return;
    const timer = setTimeout(() => void act(), speed);
    return () => clearTimeout(timer);
  }, [game, busy, paused, error, act, speed, session.replay]);
  useEffect(() => {
    if (
      !playing ||
      busy ||
      loadingLatest ||
      loadingReplay ||
      error ||
      index >= max
    ) {
      if (index >= max) setPlaying(false);
      return;
    }
    const timer = setTimeout(
      () => setIndex((i) => Math.min(i + 1, max)),
      speed,
    );
    return () => clearTimeout(timer);
  }, [playing, busy, loadingLatest, loadingReplay, error, index, max, speed]);
  return {
    game,
    busy: busy || loadingLatest || loadingReplay,
    error,
    setError,
    act,
    refresh,
    index,
    setIndex,
    max,
    playing,
    setPlaying,
    speed,
    setSpeed,
    paused,
    setPaused,
  };
}
