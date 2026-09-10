# Catanatron Game Lab

A new UI, implemented independently of `ui/`, using React 19, TypeScript, Vite,
Motion, Lucide icons, native dialogs, and a zoomable SVG board. It uses the existing
Flask API and database. Game snapshots include rule metadata for inspection.

## Use with Docker

From the **repository root**:

```sh
docker compose -f docker-compose.yml -f ui-next/compose.override.yml up --build --renew-anon-volumes
```

Open http://localhost:3000. The override changes only the UI build context and
source mount. `--renew-anon-volumes` prevents the old UI's dependencies from being
reused. The database's bind-mounted data is preserved.

Alternatively, edit both `build: ./ui` and `./ui:/app` in the `react-ui` service to
use `./ui-next`. Keep the existing port mapping and `/app/node_modules` volume.
The original UI and root Compose file are unchanged.

Source edits hot reload through Vite, including React Fast Refresh and CSS updates.
The override enables file polling for Windows/Docker Desktop bind mounts. After
changing the Compose configuration, apply it once without rebuilding:

```sh
docker compose -f docker-compose.yml -f ui-next/compose.override.yml up -d --no-deps react-ui
```

Keep using both Compose files to select the new UI. Ordinary source edits need no
restart or rebuild. After dependency or Dockerfile changes, rebuild the UI and
refresh its dependency volume:

```sh
docker compose -f docker-compose.yml -f ui-next/compose.override.yml up -d --no-deps --build --renew-anon-volumes react-ui
```

## Local development

Use Node 24 and npm:

```sh
cd ui-next
npm ci
npm start
```

The frontend defaults to the current browser hostname on port 5001 for the API.
This works on a phone visiting `http://YOUR_COMPUTER_IP:3000`, provided ports 3000
and 5001 are reachable on your local network. `localhost` on a phone means the
phone itself; use the computer's LAN address.

Override the API origin with `CTRON_API_URL` (compatible with the existing UI) or
`VITE_API_URL`, for example in `.env.local`:

```dotenv
CTRON_API_URL=http://localhost:5001
```

Restart Vite after changing environment variables. Production builds embed that
value, so set it before building. A remote HTTPS frontend needs an HTTPS API.

## Playing

- Choose Base, Mini, or Tournament; 2–4 seats; at most one human. The web API's
  supported opponents are Catanatron (alpha-beta), Weighted Random, and Random.
- Set the victory target, discard threshold, and Friendly Robber rule.
- During setup, click a glowing intersection or road. Later, select a build type
  and then a highlighted location. Pinch/scroll to zoom; drag to pan.
- The toolbar exposes legal bank/port trades and all four playable development
  cards. Robber placement offers a victim choice when more than one is legal.
- Discard one resource at a time, with the remaining count shown after each move.
- Pause/resume bot progression and adjust the delay between bot moves.
- Players, Activity, and Analysis share one tabbed sidebar; mobile uses the same
  panels via the top navigation. The larger board uses resource icons and colors,
  with names in the legend and hands. Tap a tile for its resource/number.
- Ports are separate circular resource/ratio markers connected to their two
  eligible coastal intersections, using the engine's port-node mapping.
- Hands show individual resource cards and scroll horizontally when full. Desktop
  play fits the viewport, with scrolling inside the sidebar and hands. The action
  dock keeps a fixed height and shows only legal build, trade, and card actions
  in one horizontally scrollable row. Recent moves appear in the dock, away from
  the map. The desktop workspace is capped at 1,440px with centered side gutters.
- Use the game settings button in the header to inspect the victory target,
  discard threshold, Friendly Robber rule, and player counts. Older JSON files
  without rule metadata show “Not recorded.” Ore uses the Stone icon.

Hands remain open, matching the existing UI. Multiplayer accounts, hidden hands,
and custom CLI-only bot classes are not exposed by the current web API.

## Inspecting CLI games

### JSON export: final board and complete action log, offline

```sh
catanatron-play --players W,W,R --num 1 --output games --output-format json
```

Choose **Import JSON** and select a generated game file. The current CLI exports
one final snapshot, not every historical board. The UI shows the final board and
complete activity log, and explicitly labels it as a snapshot. It does not invent
intermediate states. Imported files stay in the browser; no upload is performed.

For offline timelines, import an array of GameEncoder snapshots, `{ "states": [...] }`,
or multiple snapshot files from the same game. Frames are sorted by `state_index`.
Duplicate move numbers and inconsistent maps/action histories are rejected. The
bundled **Explore a replay** example contains 81 real engine snapshots.

### Database recording: full timeline and position analysis

Set `DATABASE_URL` to the **same database used by the web API**, then run:

```sh
catanatron-play --players W,W,R --num 1 --step-db
```

Open the printed replay link or paste the game ID into **Inspect**. Existing links
remain supported:

- `/games/:gameId` — live game
- `/replays/:gameId` — replay timeline
- `/games/:gameId/states/:stateIndex` — inspect a particular saved move

Replay controls include play/pause, first/previous/next/last, a scrubber, numeric
seek, and playback speed. With imported sparse snapshots, timeline positions
refer to frame numbers; the activity log identifies the recorded moves.
The download button exports the displayed server snapshot or the imported local
timeline. It does not download every server state.

Analysis runs on the server for the exact displayed move. It is a rough estimate:
the existing engine runs 100 simulations and divides non-current-player wins
equally among opponents. Imported offline files cannot request server analysis.

## Motion and interaction

Roads draw into place; settlements/cities spring in; the robber travels between
tiles; producing tiles pulse after a roll; individual resource cards fly into
hands on gains and animate out when spent. Production flights start at a producing
tile. Each action has an
animated event message; development-card events, turn changes, and victory also
receive feedback. Replay steps use the same transitions. The OS reduced-motion
preference is respected.

Mutations are serialized and never automatically retried. If a response is lost,
the UI pauses and asks you to reload before another move, since the server may
already have applied it. Replay reads cancel obsolete requests on seeking.

## Validation and maintenance

```sh
npm test
npm run build
```

Tests cover actual CLI exports, geometry, trades, development cards, robber victim
selection, read-only replay behavior, stale responses, and duplicate submissions.

Optional engine/API smoke check, from the repository root with `.[web]` installed:

```sh
python ui-next/scripts/smoke-api.py
```

This uses an isolated in-memory database. To regenerate the bundled example:

```sh
python ui-next/scripts/generate-example.py
```

`src/model.ts` owns the wire format, validation, and geometry; `src/api.ts` owns
HTTP; `src/useGame.ts` coordinates requests and playback. `Board.tsx`,
`ActionDock.tsx`, `Resources.tsx`, and `Panels.tsx` are presentation components;
`GameSettings.tsx` displays the recorded game rules.
The entire application is contained in this folder, with no imports from `ui/`.
