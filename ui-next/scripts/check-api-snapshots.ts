import { readFileSync } from "node:fs";
import { parseReplay, actionLabel, recordLabel } from "../src/model";
const raw = JSON.parse(
  readFileSync(new URL("../.api-smoke.json", import.meta.url), "utf8"),
);
for (const value of raw) {
  const state = parseReplay(JSON.stringify(value))[0];
  state.current_playable_actions.forEach(actionLabel);
  state.action_records.forEach(recordLabel);
}
console.log(`Frontend accepted and labeled ${raw.length} real API responses.`);
