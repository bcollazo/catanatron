# ADR: `ObservationAgent` / `PerspectivePlayer` / `Observation` — imperfect-information play

- **Status:** Proposed
- **Scope:** `catanatron` (Python >=3.11, GPL-3.0)
- **Related code:**
  - `catanatron/models/player.py` — `Player` (l.19), `decide` (l.37), `reset_state` (l.47)
  - `catanatron/models/enums.py` — `Action` (l.113), `ActionType` (l.68), `ActionPrompt` (l.58), `ActionRecord` (l.126)
  - `catanatron/state.py` — `State`, `player_state`, `action_records` (l.124)
  - `catanatron/state_functions.py` — `player_key` (l.74)
  - `catanatron/features.py` — `player_features` (l.53), `resource_hand_features` (l.85), `create_sample` (l.517), `create_sample_vector` (l.524)
  - `catanatron/game.py` — `Game.play` (l.132), `play_tick` (l.153), `execute` (l.179), `copy` (l.214), `is_valid_action` (l.22)
  - `catanatron/apply_action.py` — rewrites of `Action.value` that leak hidden info (l.254, l.316, l.266)
  - `catanatron/models/board.py` — `Board` (l.39)
  - `catanatron/cli/cli_players.py` — `register_cli_player` (l.86), `parse_cli_string` (l.70)
  - `catanatron/cli/play.py` — `--code` module loading (l.181)
  - `tests/` — repo test root; new `tests/test_observation.py`

## 1. Context & motivation

Catanatron's `Player.decide(game, playable_actions)` hands every bot the **full-truth `Game`**. That is correct for self-play, training, and offline analysis — but it makes the shipped bots (MCTS, AlphaBeta, ...) **cheaters** against real humans: a bot that can read `state.player_state["P1_WOOD_IN_HAND"]` or an opponent's hidden `ACTUAL_VICTORY_POINTS` has an information advantage no human has.

We want an opt-in, backward-compatible layer so a bot can play against humans *fairly* — it must only perceive what a real player can observe. The engine keeps full truth for simulation.

**Core principle.** An `Observation` is the bot's **entire information world**. It is *not necessarily* a projection of a perfect-information `Game`:

- **Local / engine mode:** the engine holds full truth; `PerspectivePlayer` projects it down to what is fair. This is where testing, self-play, sampling, and future search-under-uncertainty live.
- **Deployed / site mode:** the bot is a client of an external Catan site. There is **no `Game` object anywhere** — the only data is what the site feeds (own hand, public board, public events). The observation *is* the representation.

The whole design therefore converges on one seam: **bot logic may depend only on the `Observation` surface**, never on `Game`. The same agent runs against a local engine or a future site adapter that feeds observations from external events.

The current `Player.decide` conflates *perception* (what the agent is told) with *simulation* (the ability to `game.copy()` + `execute()` for search). This split uncouples them.

## 2. Decision

Add three plain classes with **zero breaking changes** to the existing engine:

- `ObservationAgent` — a pure, deployment-agnostic agent base. Decides from an `Observation`, never a `Game`. Not a `Player`.
- `PerspectivePlayer(Player)` — the engine-side **adapter**. Wraps an `ObservationAgent`; on every decision it projects the perfect-information `game` down to the agent's color as an `Observation` and forwards. This is the only engine-coupled piece.
- `Observation` — a **pure-data, read-only snapshot** of exactly one color's information world. Holds no reference to `Game` or `State`.

The game loop always requires a perfect-information state, so a fair agent *must* be wrapped by a `Player` subclass that converts the `game` parameter into an observation. The engine (`Player`, `Game`, `State`, `Action`, `generate_playable_actions`, `GameAccumulator`) is untouched; the split exists only at the decision boundary.

```python
# catanatron/models/observation_agent.py
class ObservationAgent:
    """Base class for agents that only ever perceive Observations."""

    def __init__(self, color):
        self.color = color

    def decide_observation(self, observation, playable_actions):
        raise NotImplementedError

    def reset_state(self):
        """Hook for resetting memory between games."""
        pass


# catanatron/models/perspective_player.py
class PerspectivePlayer(Player):
    """Player that adapts an ObservationAgent into the perfect-info game loop."""

    def __init__(self, agent, is_bot=True):
        self.agent = agent
        super().__init__(agent.color, is_bot)

    def decide(self, game, playable_actions):
        color = self.agent.color
        observation = Observation(
            color=color,
            features=create_sample(game, color),
            public_history=_sanitize_history(game, color),
            current_prompt=game.state.current_prompt,
            current_trade=game.state.current_trade,
            acceptees=game.state.acceptees,
            public_state=_build_public_state(game),
        )
        return self.agent.decide_observation(observation, playable_actions)

    def reset_state(self):
        self.agent.reset_state()
```

`reset_state()` lives on the agent (no-op hook, game-agnostic) and is delegated to by the adapter. It is *not* inherited from `Player`.

**File placement.** Six files in `catanatron/models/`:

| File | Contents |
|---|---|
| `player.py` | `Player`, `SimplePlayer`, `HumanPlayer`, `RandomPlayer` — **untouched** |
| `observation.py` | `Observation` — pure data, no engine imports |
| `observation_agent.py` | `ObservationAgent` — pure agent, no engine imports |
| `trade.py` | `TradeOffer` + `PendingTrades` — typed, pure-data trade snapshots |
| `public_state.py` | `PublicState`/`PublicBoard`/`PublicMap`/`PublicPlayer` — typed, pure-data public snapshots |
| `inventory.py` | `Inventory` — typed, pure-data private-hand snapshot |
| `perspective_player.py` | `PerspectivePlayer(Player)` + the `public_history` sanitizer + the `public_state`/`trade`/`inventory` projectors (`_build_public_state`, `_build_pending_trades`, `_build_inventory`) |

All new classes are plain classes (no `Protocol`/`ABC`), matching the repo's existing style. The import graph is acyclic: `observation.py` / `observation_agent.py` depend only on `models.player` (stdlib-only); `perspective_player.py` → `observation.py` + `features.py`/`game.py` + `models.player`. Only `perspective_player.py` couples to the engine.

## 3. `Observation` API

```python
# catanatron/models/observation.py
@dataclass(frozen=True)
class Observation:
    color: Color
    features: Optional[Dict[str, Any]] = None  # optional; RL-vector surface only
    public_history: Tuple[ActionRecord, ...] = ()
    current_prompt: Optional[ActionPrompt] = None
    pending_trades: PendingTrades = PendingTrades()
    public_state: Optional[PublicState] = None
    inventory: Optional[Inventory] = None

    @property
    def own(self):  # the observer's own PublicPlayer, for obs.own.public_vps
        return self.public_state.players[self.color] if self.public_state else None
```

`Observation` is constructed with precomputed data and is immutable by convention. The extraction from the full-truth `Game` happens **in `PerspectivePlayer`**, eagerly, once per decision.

### 3.1 Tiers

**`own` — private (yours):** hand counts per resource, dev cards in hand, own `ACTUAL_VICTORY_POINTS` (you know your own hidden VP cards; carried on `Observation.inventory.actual_vps`), own buildings, buildable nodes/edges, `has_rolled`, ports owned.

**`public` — known with certainty, including derived counts:** board (buildings map, roads, robber, `CatanMap` tiles/numbers/adjacency/production/ports), per-player `public_vps`, `has_road`, `has_army`, `longest_road_length`, `roads/settlements/cities_left`, `knights_played`, `num_resources` and `num_devs` hand/dev-card **counts** (public in real Catan; already exposed by `resource_hand_features`), and a filtered `public_history` (below).

**`revelations` — composition-evidence only (for belief-state):** discards on a 7 (count public), steals `(stealer, victim)` (card identity hidden), dev-card purchases (count public, identity hidden), trades with exact cards (fully public).

**Never exposed:** opponent `*_IN_HAND` per-resource/per-dev fields, opponent `ACTUAL_VICTORY_POINTS`, stolen/discarded card identities for non-participants, which dev card an opponent drew.

### 3.2 `features` — projection via the existing RL extractors

The flat feature dict is built by reusing `features.create_sample(game, p0_color)`, which is *already* an imperfect-information-shaped snapshot: `P0_ACTUAL_VPS` only for yourself, `P0_*_IN_HAND` only for yourself, per-player public counts (`P{i}_NUM_RESOURCES_IN_HAND`, `P{i}_NUM_DEVS_IN_HAND`, `P{i}_*_PLAYED`, `P{i}_PUBLIC_VPS`, board features), all keyed relative to `P0` = the observing color.

`create_sample` is called by `PerspectivePlayer.decide` and the resulting dict is stored on `Observation.features`. It is **eager** (computed once per real decision) rather than lazy: this is what allows `Observation` to hold no `Game` reference. The cost is negligible — `create_sample` runs at decision time, never inside the MCTS/playout hot path (which uses `game.copy()`).

`features` is optional on the `Observation` (defaults to `None`): it exists for compatibility with the RL training loop and is not part of the agent's core information world (`public_state` + `pending_trades` + `inventory` + `current_prompt` carry the same facts in typed form).

### 3.2.1 `public_state` — structured access to public state

`features` is flat, P0-relative, and schema-encoded — fine for RL vectors, hostile to an agent that wants to reason about *where* things are. The engine gives `Player` a structured `Board` (`game.state.board.buildings`, `.roads`, `.robber_coordinate`) that the `ObservationAgent` has no access to. To close that gap, `Observation.public_state` carries a pure-data snapshot of **every public fact in `State`**, keyed absolutely (node/edge ids and `Color`), built by `_build_public_state(game)` in `PerspectivePlayer`:

```python
public_state = {
    "board": {
        "buildings": {node_id: (color, building_type)},  # current buildings
        "roads": {(n1, n2): color},                      # canonical orientation
        "robber_coordinate": (x, y, z),
        "longest_road_color": color | None,
        "longest_road_length": int,
    },
    "players": {  # one entry per color, public facts only
        Color.RED: {
            "public_vps": int, "has_army": bool, "has_road": bool,
            "longest_road_length": int, "roads_left": int,
            "settlements_left": int, "cities_left": int, "has_rolled": bool,
            "hand_resource_count": int, "hand_dev_count": int,
            "played_knight": int, "played_monopoly": int,
            "played_road_building": int, "played_year_of_plenty": int,
            "played_victory_point": int,
        },
        ...
    },
}
```

Invariants mirror the tiers (§3.1): only **public** facts appear — opponent hand identities and `ACTUAL_VICTORY_POINTS` are never projected. Like `features`, it is eager (once per decision) and pure data (no `Game`/`State` reference, so the fairness contract holds). The static `CatanMap` (tiles/numbers/adjacency/ports) is also snapshotted here as `public_state.board.map` (`PublicMap`: tiles by id with resource+roll, ports by id with resource+node ids, per-node `adjacent_tiles`, and `land_nodes`), giving fair agents structured, absolute-keyed access to the terrain instead of only the flat `TILE*`/`PORT*` feature keys. Derived probabilities are **not** stored — per-node production is inferred from tile rolls plus adjacency — keeping the snapshot raw facts only. It is static, so every decision snapshots identical terrain; the projection exists solely to keep `Observation` free of engine references.

### 3.3 `public_history` — sanitized action log

`State.action_records` is full truth. The engine **rewrites `Action.value`** for some actions, so hidden identities sit in `action.value`, not just `ActionRecord.result`:

- `apply_buy_development_card` → `Action(..., card)` (apply_action.py:254)
- `apply_discard` → `Action(..., discarded_resource)` (apply_action.py:316)
- `apply_roll` → `Action(..., dice)` — dice are public (apply_action.py:266)
- `apply_move_robber` → `action.value = (coordinate, robbed_color)`, `result = robbed_resource`

Therefore sanitization is **keyed on `action_type` plus participation**, never on "is `result` non-None". It lives in `perspective_player.py` (`_sanitize_history`/`_sanitize_record`):

| Opponent record | `public_history` exposes | Reasoning |
|---|---|---|
| `BUY_DEVELOPMENT_CARD` | purchase only; `value` and `result` redacted to `None` | identity is the private fact; redact both channels |
| `MOVE_ROBBER` | `value` `(coordinate, robbed_color)`; `result` redacted to `None` **unless observer is the victim** | participants know the card, spectators do not |
| `DISCARD_RESOURCE` | full identity | public per tournament convention (see assumption below) |
| everything else | pass through unchanged | rolls, builds, trades are public |

For the observer's own records, full detail is retained (it is their own private information; the fairness contract is unaffected).

**Assumption (documented):** discards are treated as **public** — the official rules/FAQ do not state discard privacy; physically cards return to the face-up supply, and tournament play treats discards as public. If a future deployment targets house rules with face-down discards, this is a single sanitizer switch (`discards_public=True`).

### 3.4 Trade representation

Trades are **completely public**: `OFFER_TRADE`'s 10-tuple, `state.current_trade`, `acceptees`, and `CONFIRM_TRADE`'s accepting color carry zero hidden info. Decisions are already fully encoded in `playable_actions` (`ACCEPT_TRADE`/`REJECT_TRADE` embed `current_trade` in `.value`; `CONFIRM_TRADE` embeds offer + acceptor).

Rather than expose the engine's raw positional `current_trade` tuple plus a parallel `acceptees` tuple, `Observation` carries `pending_trades: PendingTrades`, an immutable container for the trades on the table. Each `TradeOffer` re-keys that info structurally:

```python
@dataclass(frozen=True)
class TradeOffer:
    offerer: Color
    offered: Dict[FastResource, int]   # resource -> count given up
    asking: Dict[FastResource, int]    # resource -> count wanted
    acceptees: Dict[Color, bool]       # color -> has accepted (offerer always False)

@dataclass(frozen=True)
class PendingTrades:
    offers: Tuple[TradeOffer, ...] = ()
    # iterable/indexable like a tuple; frozen so the set of offers is read-only
    # convenience accessors: .is_active, .single (the lone offer in engine mode)
```

The engine's `State` holds a single `current_trade` slot (`apply_action.py:441-515`), so today the container holds zero or one offer; the container future-proofs site-mode rules that allow competing offers simultaneously. Built by `_build_pending_trades` in `perspective_player.py`, which reads the last `OFFER_TRADE` record for the offerer (the engine stores a turn *index*, not the color).

`Observation` also carries `current_prompt` so bots need not dig through `playable_actions` to know what is being asked of them. Generating your own `OFFER_TRADE` needs `has_rolled`, available in the `own` tier via `features["P0_HAS_ROLLED"]`.

### 3.5 Presentation-free

`Observation` contains **no `describe`/`render` logic** — it is a lean, data-only snapshot. A standalone text helper (e.g. `observation_to_text(obs)`) may be added later for LLM and `HumanPlayer` consumers.

### 3.6 Hard boundary

`Observation` holds **no reference to `Game` or `State`** — not even privately, and not "a copy for simulation." This is what makes the fairness contract structural: there is no handle to leak. A future belief-state hook must build a *fresh* complete game from sampled hidden info, never from the observation.

## 4. Integration map

All seams work unchanged:

```python
# Library
game = Game(
    [
        PerspectivePlayer(CardCounterBot(Color.RED)),
        HumanPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
    ]
)
winner = game.play()

# CLI (register_cli_player) → catanatron-play --code myfile.py --players CC,R,R,R
from catanatron.cli.cli_players import register_cli_player
register_cli_player("CC", lambda color: PerspectivePlayer(CardCounterBot(color)))

# Ad-hoc decide_fn seam (play_tick calls decide_fn(player, game, playable_actions))
# drives bare ObservationAgents directly, no wrapper needed:
game.play(decide_fn=lambda agent, g, actions: agent.decide_observation(
    Observation(
        color=agent.color,
        features=create_sample(g, agent.color),
        public_history=_sanitize_history(g, agent.color),
        current_prompt=g.state.current_prompt,
        pending_trades=_build_pending_trades(g),
        public_state=_build_public_state(g),
    ),
    actions,
))

# reset_state() delegated from PerspectivePlayer to its agent
```

`json.py` / replay / accumulators are **untouched**: full-truth serialization is the engine's business (replay reconstructs hidden state from `action_record.result`). This ADR does not change it.

## 5. Fairness contract

Unit-testable invariants:

1. **Structural:** an `ObservationAgent` never receives a `Game` or `State` — only `Observation` (type-enforced at the only `game`-touching site, `PerspectivePlayer.decide`). The agent is *not* a `Player` and exposes no `decide(game, ...)` / `is_bot`.
2. **Counts reachable, identities not:** for every opponent, resource/dev-card hand *counts* are reachable from the observation surface; per-resource/per-dev hand *identities* are not.
3. **Hidden fields absent:** opponent `ACTUAL_VICTORY_POINTS`, non-participant stolen-card identities, and non-buyer drawn-dev-card identities are absent from the entire exposed surface; `Observation` carries no `_game`/`state` attribute at all.
4. **`public_state`/`pending_trades` are public-only:** the structured snapshots mirror the engine's board/per-player public fields exactly (§3.2.1) and the typed trade offers; both are scanned by the leak-invariant test alongside `features`.

`tests/test_observation.py` coverage:
- **Structural separation tests:** `ObservationAgent` is not a `Player`; `Observation` is pure data with no `_game`/`state`.
- **Leak-invariant test** (`scan` over `features` + `public_state` + `public_history` across seeded full games): forbids opponent `*_IN_HAND`, opponent `ACTUAL_VICTORY_POINTS`, opponent `BUY_DEVELOPMENT_CARD` value/result, and (for spectators) `MOVE_ROBBER` stolen-card identities.
- **`public_state` surface tests:** each decision's snapshot exactly matches the engine board (`buildings`, canonical `roads`, `robber_coordinate`, longest road, static `map` tiles/ports/adjacency) and per-player public counts, verified against the state it was projected from.
- **Participant test:** the victim of a steal *does* see the stolen card in `public_history`; a spectator does not.
- **Sanitizer unit tests** per row of the table in §3.3.
- **Seam tests:** `Game([PerspectivePlayer(bot), RandomPlayer, ...])` plays to completion; bare `ObservationAgent`s driven through the `decide_fn` seam; `reset_state()` delegation.

The invariant test scans the **output surface** (not implementation), so it stays valid if projection internals change.

**Deferred:** a "fair bot vs full-info bot" win-rate benchmark. It is meaningless with a trivial test bot and belongs to a follow-up once a real fair bot exists.

## 6. Performance

- Constructing an `Observation` is O(1) — it stores precomputed data and a color.
- `public_state` projection is O(board + players) per decision and builds only plain dicts/tuples — negligible next to `create_sample`.
- `features` is computed **once per decision**, eagerly, in `PerspectivePlayer.decide`. The MCTS/playout hot path — which performs thousands of `copy()`s — never constructs an `Observation`.
- `public_history` sanitization runs once per decision and is O(history length); it is cheap enough per decision.

## 7. Consequences & tradeoffs

- **Additive API surface.** Existing `Player`-based code paths are untouched and still work; perfect bots remain full-information by design.
- **Structural fairness** replaces "trust the bot to be honest." An `ObservationAgent` cannot receive the `Game` — the only `game`-touching site is the `PerspectivePlayer` adapter, and `Observation` carries no engine handle — so the fair contract is enforceable, not a promise.
- **Belief-state / search-under-uncertainty (contract defined, not implemented).** A future hook `ObservationAgent.sample_hidden_state(obs) -> Game` will let fair agents run the shipped perfect bots (MCTS, AlphaBeta) on their **own sampled hypotheses** of hidden state. Invariant: it returns a fresh, consistent game built only from the observation — never the wrapped `_game`. Deliberately deferred: the construction plumbing is non-trivial and there is no v1 consumer.
- **`HumanPlayer` unchanged (out of scope).** A human cannot read the `Game` object — they already only perceive their own hand and the board — so there is no fairness benefit in re-plumbing them. Flagged as a future adapter target.
- **Deployed-play implications.** On a real Catan site there is no perfect-information representation; game "replays" are observation logs, not full-truth. Because `Observation` is pure data, a site adapter can build the *same* object from external events and drive the *same* `ObservationAgent`. Future work: an **observation-log serializer** (record `features` + `public_history` per decision for analysis/ML on real-site games) and a **site adapter** that feeds external events through `decide_observation`. The framework's contract (§5) is what makes both buildable later without redesign.
- **Assumption on discards.** Treated as public (tournament convention). A future house-rules deployment can flip a single sanitizer switch.

## 8. Resolved open questions

| # | Question | Decision |
|---|---|---|
| 1 | Module placement / Protocol-ABC? | Four files: `models/player.py` (untouched), `models/observation.py`, `models/observation_agent.py`, `models/perspective_player.py`. Plain classes, no Protocol/ABC. Only `perspective_player.py` imports the engine. |
| 2 | Is `PerspectivePlayer` a subclass of `Player`? | **No** for the agent, **yes** for the adapter. `ObservationAgent` is deliberately *not* a `Player` (it is not-a full-truth decider, it is engine-independent, and it carries no `decide(game, ...)`); `PerspectivePlayer` *is* a `Player` because the game loop requires a perfect-info state and the adapter's job is to convert it into an observation. |
| 3 | Where does `create_sample` live? | In `PerspectivePlayer` (eager, once per decision), not on `Observation` — so `Observation` is pure data and free of `Game`/`State`. |
| 4 | `public_history` filtering? | Keyed on `action_type` + participant (§3.3). Redact both `action.value` and `result` for opponent `BUY_DEV`; redact `result` for `MOVE_ROBBER` unless victim; discards public. Lives in `perspective_player.py`. |
| 5 | Belief-state hook now or defer? | Contract in this ADR only; zero v1 implementation; future work. |
| 6 | Trade sub-prompts / describe-render? | Trades fully public; `Observation` carries `current_prompt` + typed `pending_trades: PendingTrades` (immutable container of `TradeOffer`, each re-keyed by resource and color). No `describe`/render in `Observation`. The engine allows one offer at a time today; the container future-proofs competing offers. |
| 7 | `reset_state()`? | Lives on the agent as a no-op hook; `PerspectivePlayer.reset_state` delegates. Not inherited from `Player`. |
| 8 | `HumanPlayer` perspective view? | Out of scope for v1; future adapter candidate. |
| 9 | Test strategy? | `tests/test_observation.py`: structural separation + leak-invariant surface scan (features + public_state + pending_trades + public_history) + participant + sanitizer-table + trade-object + seam tests; win-rate benchmark deferred. |
| 10 | `json.py`/replay/accumulators? | Untouched. Observation-log serialization and site adapter are future work. |
| 11 | Structured public-state access for agents? | `Observation.public_state`: a pure-data, absolute-keyed snapshot of all public facts (board buildings/roads/robber/longest road + per-color public counts + the static map as `board.map`), built eagerly by `_build_public_state` in `perspective_player.py`. Complements the P0-relative flat `features`; the `CatanMap` is included as a structured `PublicMap` (raw tile rolls/ports/adjacency, no derived probabilities) so agents can reason about the terrain, not just parse `TILE*`/`PORT*` keys. |
