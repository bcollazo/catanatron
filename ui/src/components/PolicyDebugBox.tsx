import { useContext, useMemo, useState } from "react";
import { Alert, Divider, MenuItem, Select, SelectChangeEvent } from "@mui/material";

import { store } from "../store";

import "./PolicyDebugBox.scss";

type PolicyDebugBoxProps = {
  stateIndex: number;
};

const TOP_N_OPTIONS = [3, 5, 8, 10, 15];

/** Softmax tails are often tiny on the full 245-simplex; extra precision avoids fake zeros. */
function formatPolicyMassPercent(p: number): string {
  const pct = p * 100;
  if (!Number.isFinite(p)) return "—";
  if (pct <= 0) return "0%";
  if (pct < 1e-6) return `${pct.toExponential(2)}%`;
  if (pct < 0.01) return `${pct.toFixed(5)}%`;
  if (pct < 1) return `${pct.toFixed(4)}%`;
  if (pct >= 99.9999) return `${pct.toPrecision(8)}%`;
  return `${pct.toFixed(3)}%`;
}

export default function PolicyDebugBox({ stateIndex }: PolicyDebugBoxProps) {
  const { state } = useContext(store);
  const [topN, setTopN] = useState<number>(5);

  const policyState = useMemo(() => {
    const records = state.gameState?.policy_debug_records;
    if (records == null || records.length === 0) {
      return { kind: "missing" as const };
    }
    const entry = records[Math.max(0, records.length - 1)];
    if (entry === null || entry === undefined) {
      return { kind: "noop" as const };
    }
    return { kind: "ok" as const, entry };
  }, [state.gameState, stateIndex]);

  const debugEntry = policyState.kind === "ok" ? policyState.entry : null;

  const selectedSummary = useMemo(() => {
    if (!debugEntry) return "N/A";
    if (debugEntry.chosen_action_description_detailed) {
      return `#${debugEntry.chosen_action_index} — ${debugEntry.chosen_action_description_detailed}`;
    }
    const chosen = debugEntry.top_actions.find(
      (x) => x.action_index === debugEntry.chosen_action_index
    );
    const desc = chosen?.description_detailed || chosen?.description || "unknown";
    return `#${debugEntry.chosen_action_index} — ${desc}`;
  }, [debugEntry]);

  const rowModels = useMemo(() => {
    if (!debugEntry) return [];
    const slice = debugEntry.top_actions.slice(0, topN);
    const masses = slice.map((r) =>
      r.probability_given_valid != null ? r.probability_given_valid : r.probability
    );
    const massSum = masses.reduce((s, x) => s + x, 0);
    const denom = massSum > 0 ? massSum : 1;
    const hasLegalField = slice.some((r) => r.probability_given_valid != null);

    return slice.map((row, i) => ({
      row,
      rank: i + 1,
      pctAmongListed: (masses[i] / denom) * 100,
      hasLegalField,
    }));
  }, [debugEntry, topN]);

  const chosenLegal =
    debugEntry?.chosen_action_probability_given_valid != null
      ? debugEntry.chosen_action_probability_given_valid
      : null;

  return (
    <div className="policy-debug-box">
      <div className="header">
        <strong>Policy Debug</strong>
        <Select
          size="small"
          value={String(topN)}
          onChange={(e: SelectChangeEvent<string>) => setTopN(Number(e.target.value))}
        >
          {TOP_N_OPTIONS.map((n) => (
            <MenuItem key={n} value={String(n)}>
              Top {n}
            </MenuItem>
          ))}
        </Select>
      </div>
      <p className="policy-debug-legend">
        <strong>Legal</strong> = probability among currently legal moves only (sums to 100% over all legal
        indices). <strong>Global</strong> = raw softmax mass on the full 245-vector (invalid slots ≈ 0).
        Re-record replays after updating the trainer to populate Legal; older logs only have Global.
      </p>
      {policyState.kind === "missing" && (
        <Alert severity="info">
          No policy debug metadata on this replay step. Older replay files do not
          include per-step policy analytics.
        </Alert>
      )}
      {policyState.kind === "noop" && (
        <Alert severity="info">
          No policy debug for this step (opponent or environment action).
        </Alert>
      )}
      {debugEntry && (
        <>
          <div className="meta">
            <div>
              <span>State value:</span> {debugEntry.state_value_estimate.toFixed(4)}
            </div>
            <div className="chosen">
              <span>Chosen action:</span> {selectedSummary}
            </div>
            <div className="chosen-prob">
              {chosenLegal != null ? (
                <>
                  <span>P(chosen | legal):</span> {formatPolicyMassPercent(chosenLegal)}{" "}
                  <span className="muted">
                    (global {formatPolicyMassPercent(debugEntry.chosen_action_probability)})
                  </span>
                </>
              ) : (
                <>
                  <span>P(chosen):</span> {formatPolicyMassPercent(debugEntry.chosen_action_probability)}{" "}
                  <span className="muted">(re-export replay for Legal column)</span>
                </>
              )}
            </div>
          </div>
          <Divider />
          <div className="policy-table-head" aria-hidden>
            <span>#</span>
            <span>Legal</span>
            <span>Global</span>
            <span>Top‑{topN}′</span>
            <span>Action</span>
          </div>
          <div className="rows">
            {rowModels.map(({ row, rank, pctAmongListed, hasLegalField }) => {
              const pLegal = row.probability_given_valid;
              const showLegal = pLegal != null;
              return (
                <div className="row" key={`${row.action_index}-${rank}`}>
                  <span className="rank">{rank}</span>
                  <span
                    className="prob prob-legal"
                    title="Renormalized over legal moves — sums to 100% across all legal actions"
                  >
                    {showLegal ? formatPolicyMassPercent(pLegal) : "—"}
                  </span>
                  <span
                    className="prob prob-global"
                    title="Absolute softmax mass on full action vector"
                  >
                    {formatPolicyMassPercent(row.probability)}
                  </span>
                  <span
                    className="prob-among"
                    title={
                      hasLegalField
                        ? "Share among listed rows using Legal masses"
                        : "Share among listed rows using Global masses"
                    }
                  >
                    {pctAmongListed.toFixed(2)}%
                  </span>
                  <div className="desc-block">
                    <div className="desc-main">{row.description_detailed || row.description}</div>
                    <div className="desc-meta">index {row.action_index}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
