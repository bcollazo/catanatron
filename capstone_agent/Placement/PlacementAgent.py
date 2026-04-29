import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

try:
    from ..CONSTANTS import FEATURE_SPACE_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE
    from ..PPOHyperparams import PPOHyperparams
    from .PlacementGNNModel import PlacementGNNModel
    from .PlacementModel import PlacementModel
    from .placement_heuristic import (
        DEFAULT_HEURISTIC_CONFIG,
        sample_from_scores,
        score_legal_roads,
        score_legal_settlements,
    )
    from ..RolloutBuffer import RolloutBuffer
    from ..device import get_device
    from .placement_action_space import (
        PlacementPrompt,
        capstone_action_to_local,
        capstone_mask_to_local_mask,
        infer_placement_prompt,
        local_action_size,
        local_action_to_capstone,
    )
    from .placement_features import (
        COMPACT_NODE_FEATURE_SIZE,
        COMPACT_PLACEMENT_FEATURE_SIZE,
        NUM_NODES,
        STEP_INDICATOR_SIZE,
        project_capstone_batch_to_compact_placement,
        project_capstone_to_compact_placement,
    )
except ImportError:  # pragma: no cover - supports script-style imports
    from CONSTANTS import FEATURE_SPACE_SIZE, PLACEMENT_AGENT_HIDDEN_SIZE
    from PPOHyperparams import PPOHyperparams
    from PlacementGNNModel import PlacementGNNModel
    from PlacementModel import PlacementModel
    from placement_heuristic import (
        DEFAULT_HEURISTIC_CONFIG,
        sample_from_scores,
        score_legal_roads,
        score_legal_settlements,
    )
    from RolloutBuffer import RolloutBuffer
    from device import get_device
    from placement_action_space import (
        PlacementPrompt,
        capstone_action_to_local,
        capstone_mask_to_local_mask,
        infer_placement_prompt,
        local_action_size,
        local_action_to_capstone,
    )
    from placement_features import (
        COMPACT_NODE_FEATURE_SIZE,
        COMPACT_PLACEMENT_FEATURE_SIZE,
        NUM_NODES,
        STEP_INDICATOR_SIZE,
        project_capstone_batch_to_compact_placement,
        project_capstone_to_compact_placement,
    )

# ── registry of available strategies ─────────────────────────────

PLACEMENT_STRATEGIES = {}


def register_strategy(name):
    """Decorator that adds a class to the PLACEMENT_STRATEGIES dict."""
    def wrapper(cls):
        PLACEMENT_STRATEGIES[name] = cls
        return cls
    return wrapper


def make_placement_agent(strategy: str = "model", **kwargs):
    """Construct a placement agent by strategy name.

    Args:
        strategy: One of the keys in PLACEMENT_STRATEGIES
                  (currently ``"model"`` or ``"random"``).
        **kwargs: Forwarded to the chosen class constructor.

    Returns:
        An agent that implements select_action / store / train / load / save.
    """
    if strategy not in PLACEMENT_STRATEGIES:
        available = ", ".join(sorted(PLACEMENT_STRATEGIES))
        raise ValueError(
            f"Unknown placement strategy {strategy!r}. Choose from: {available}"
        )
    return PLACEMENT_STRATEGIES[strategy](**kwargs)


# ── random baseline ──────────────────────────────────────────────

@register_strategy("random")
class RandomPlacementAgent:
    """Picks uniformly at random from valid actions.  No model, no training."""

    def __init__(self, **_kwargs):
        pass

    def select_action(self, state, mask, **_kwargs):
        valid = np.where(np.asarray(mask) > 0.5)[0]
        if len(valid) == 0:
            raise ValueError("RandomPlacementAgent received a mask with no valid actions")
        action = int(np.random.choice(valid))
        n_valid = len(valid)
        log_prob = -math.log(n_valid) if n_valid > 0 else 0.0
        value = 0.0
        return (action, log_prob, value)

    def store(self, state, mask, action, log_prob, reward, value, done):
        pass

    def train(self, last_value):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


@register_strategy("heuristic")
class HeuristicPlacementAgent:
    """Score legal opening placements with a static Catan heuristic."""

    def __init__(self, temperature: float = 0.5, heuristic_config=None, **_kwargs):
        self.temperature = float(temperature)
        self.heuristic_config = heuristic_config or DEFAULT_HEURISTIC_CONFIG

    def select_action(self, state, mask, **kwargs):
        del state

        game = kwargs.get("game")
        if game is None:
            raise ValueError("HeuristicPlacementAgent requires a `game=` keyword argument")

        full_mask = np.asarray(mask, dtype=np.float32)
        prompt = infer_placement_prompt(full_mask)
        local_mask = capstone_mask_to_local_mask(full_mask, prompt)

        catan_map = game.state.board.map
        current_color = game.state.current_color()

        from catanatron.models.enums import SETTLEMENT

        buildings = game.state.buildings_by_color
        my_settlements = list(buildings[current_color][SETTLEMENT])
        opp_settlements = []
        for color in game.state.colors:
            if color != current_color:
                opp_settlements.extend(buildings[color][SETTLEMENT])

        valid_local_actions = [int(idx) for idx in np.where(local_mask > 0.5)[0]]
        if prompt == PlacementPrompt.SETTLEMENT:
            scores = score_legal_settlements(
                catan_map,
                valid_local_actions,
                my_settlements,
                opp_settlements,
                self.heuristic_config,
            )
        elif prompt == PlacementPrompt.ROAD:
            scores = score_legal_roads(
                catan_map,
                valid_local_actions,
                my_settlements,
                opp_settlements,
                self.heuristic_config,
            )
        else:
            raise ValueError(f"Unknown placement prompt: {prompt}")

        chosen_local_action, log_prob = sample_from_scores(scores, self.temperature)
        capstone_action = local_action_to_capstone(prompt, chosen_local_action)
        return (capstone_action, log_prob, 0.0)

    def store(self, state, mask, action, log_prob, reward, value, done):
        pass

    def train(self, last_value):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


@register_strategy("value_heuristic")
class ValueHeuristicPlacementAgent:
    """Choose opening placements by one-ply evaluation with the engine value fn."""

    def __init__(self, value_fn_builder_name=None, params=None, **_kwargs):
        self.value_fn_builder_name = value_fn_builder_name
        self.params = params

    def select_action(self, state, mask, **kwargs):
        del state

        game = kwargs.get("game")
        if game is None:
            raise ValueError("ValueHeuristicPlacementAgent requires a `game=` keyword argument")

        playable_actions = kwargs.get("playable_actions") or game.playable_actions
        full_mask = np.asarray(mask, dtype=np.float32)
        color = game.state.current_color()

        from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index
        from catanatron.players.value import get_value_fn

        value_fn = get_value_fn(self.value_fn_builder_name or "base_fn", self.params)
        best_action_idx = None
        best_value = float("-inf")

        for action in playable_actions:
            action_idx = catanatron_action_to_capstone_index(action)
            if full_mask[action_idx] <= 0.5:
                continue

            game_copy = game.copy()
            game_copy.execute(action)
            value = value_fn(game_copy, color)
            if value > best_value:
                best_action_idx = action_idx
                best_value = value

        if best_action_idx is None:
            raise ValueError("ValueHeuristicPlacementAgent found no legal placement action")

        return (best_action_idx, 0.0, best_value)

    def store(self, state, mask, action, log_prob, reward, value, done):
        pass

    def train(self, last_value):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


@register_strategy("rollout_value")
class RolloutValuePlacementAgent:
    """Evaluate each placement by greedily completing the opening in simulation."""

    def __init__(
        self,
        value_fn_builder_name=None,
        params=None,
        cache_decisions: bool = False,
        **_kwargs,
    ):
        self.value_fn_builder_name = value_fn_builder_name
        self.params = params
        self.cache_decisions = bool(cache_decisions)
        self._decision_cache = {}

    def _value_fn(self):
        from catanatron.players.value import get_value_fn

        return get_value_fn(self.value_fn_builder_name or "base_fn", self.params)

    def _enum_key(self, value):
        if value is None:
            return "NONE"
        return getattr(value, "value", str(value))

    def _hashable_action_value(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (list, tuple)):
            return tuple(self._hashable_action_value(item) for item in value)
        return getattr(value, "value", value)

    def _board_signature(self, game):
        catan_map = game.state.board.map
        production = []
        for node_id, resources in sorted(catan_map.node_production.items()):
            production.append(
                (
                    int(node_id),
                    tuple(
                        sorted(
                            (
                                self._enum_key(resource),
                                round(float(probability), 8),
                            )
                            for resource, probability in resources.items()
                        )
                    ),
                )
            )

        ports = tuple(
            sorted(
                (
                    self._enum_key(resource),
                    tuple(sorted(int(node_id) for node_id in node_ids)),
                )
                for resource, node_ids in catan_map.port_nodes.items()
            )
        )
        return tuple(production), ports

    def _opening_history_key(self, game):
        return tuple(
            (
                self._enum_key(getattr(record, "action", record).color),
                self._enum_key(getattr(record, "action", record).action_type),
                self._hashable_action_value(getattr(record, "action", record).value),
            )
            for record in game.state.action_records
        )

    def _decision_cache_key(self, game, playable_actions, action_indexer):
        playable = tuple(sorted(int(action_indexer(action)) for action in playable_actions))
        return (
            self._board_signature(game),
            self._opening_history_key(game),
            self._enum_key(game.state.current_color()),
            playable,
        )

    def _cached_decision(self, cache_key, full_mask):
        if not self.cache_decisions:
            return None

        cached = self._decision_cache.get(cache_key)
        if cached is None:
            return None

        action_idx, value = cached
        if full_mask[action_idx] <= 0.5:
            return None
        return action_idx, value

    def _store_decision(self, cache_key, action_idx, value):
        if not self.cache_decisions:
            return

        self._decision_cache[cache_key] = (int(action_idx), float(value))

    def _best_rollout_action(self, game, root_color, value_fn):
        maximizing = game.state.current_color() == root_color
        best_action = None
        best_value = float("-inf") if maximizing else float("inf")

        for action in game.playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)
            value = value_fn(game_copy, root_color)
            if maximizing and value > best_value:
                best_action = action
                best_value = value
            elif not maximizing and value < best_value:
                best_action = action
                best_value = value

        return best_action

    def _complete_opening(self, game, action, root_color, value_fn):
        rollout = game.copy()
        rollout.execute(action)

        while rollout.state.is_initial_build_phase and rollout.winning_color() is None:
            rollout_action = self._best_rollout_action(rollout, root_color, value_fn)
            if rollout_action is None:
                break
            rollout.execute(rollout_action)

        return rollout

    def _completed_opening_value(self, game, action, root_color, value_fn):
        rollout = self._complete_opening(game, action, root_color, value_fn)
        return value_fn(rollout, root_color)

    def select_action(self, state, mask, **kwargs):
        del state

        game = kwargs.get("game")
        if game is None:
            raise ValueError("RolloutValuePlacementAgent requires a `game=` keyword argument")

        playable_actions = kwargs.get("playable_actions") or game.playable_actions
        full_mask = np.asarray(mask, dtype=np.float32)
        color = game.state.current_color()

        from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index

        cache_key = self._decision_cache_key(
            game,
            playable_actions,
            catanatron_action_to_capstone_index,
        )
        cached = self._cached_decision(cache_key, full_mask)
        if cached is not None:
            action_idx, value = cached
            return (action_idx, 0.0, value)

        value_fn = self._value_fn()
        best_action_idx = None
        best_value = float("-inf")

        for action in playable_actions:
            action_idx = catanatron_action_to_capstone_index(action)
            if full_mask[action_idx] <= 0.5:
                continue

            value = self._completed_opening_value(game, action, color, value_fn)
            if value > best_value:
                best_action_idx = action_idx
                best_value = value

        if best_action_idx is None:
            raise ValueError("RolloutValuePlacementAgent found no legal placement action")

        self._store_decision(cache_key, best_action_idx, best_value)
        return (best_action_idx, 0.0, best_value)

    def store(self, state, mask, action, log_prob, reward, value, done):
        pass

    def train(self, last_value):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


@register_strategy("beam_value")
class BeamValuePlacementAgent(RolloutValuePlacementAgent):
    """Evaluate placements with a bounded minimax beam over the opening phase.

    ``rollout_value`` assumes both players greedily choose the immediate best
    value-function action for the rest of the initial build.  This variant keeps
    the same leaf evaluator, but spends a small budget considering several
    plausible opponent and self replies before falling back to greedy completion.
    """

    def __init__(
        self,
        value_fn_builder_name=None,
        params=None,
        top_k: int = 3,
        search_depth: int = 3,
        root_top_k: int | None = 16,
        leaf_rollout: bool = True,
        **_kwargs,
    ):
        super().__init__(value_fn_builder_name=value_fn_builder_name, params=params)
        self.top_k = max(1, int(top_k))
        self.search_depth = max(0, int(search_depth))
        self.root_top_k = None if root_top_k is None else max(1, int(root_top_k))
        self.leaf_rollout = bool(leaf_rollout)

    def _leaf_value(self, game, root_color, value_fn):
        if not self.leaf_rollout:
            return value_fn(game, root_color)

        rollout = game.copy()
        while rollout.state.is_initial_build_phase and rollout.winning_color() is None:
            rollout_action = self._best_rollout_action(rollout, root_color, value_fn)
            if rollout_action is None:
                break
            rollout.execute(rollout_action)
        return value_fn(rollout, root_color)

    def _ranked_children(self, game, root_color, value_fn, limit):
        maximizing = game.state.current_color() == root_color
        scored = []

        for action in game.playable_actions:
            child = game.copy()
            child.execute(action)
            scored.append((value_fn(child, root_color), child))

        scored.sort(key=lambda item: item[0], reverse=maximizing)
        return scored[:limit]

    def _search(self, game, root_color, value_fn, depth):
        if (
            depth <= 0
            or not game.state.is_initial_build_phase
            or game.winning_color() is not None
        ):
            return self._leaf_value(game, root_color, value_fn)

        maximizing = game.state.current_color() == root_color
        best_value = float("-inf") if maximizing else float("inf")

        for _, child in self._ranked_children(game, root_color, value_fn, self.top_k):
            value = self._search(child, root_color, value_fn, depth - 1)
            if maximizing:
                best_value = max(best_value, value)
            else:
                best_value = min(best_value, value)

        return best_value

    def select_action(self, state, mask, **kwargs):
        del state

        game = kwargs.get("game")
        if game is None:
            raise ValueError("BeamValuePlacementAgent requires a `game=` keyword argument")

        playable_actions = kwargs.get("playable_actions") or game.playable_actions
        full_mask = np.asarray(mask, dtype=np.float32)
        color = game.state.current_color()

        from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index

        cache_key = self._decision_cache_key(
            game,
            playable_actions,
            catanatron_action_to_capstone_index,
        )
        cached = self._cached_decision(cache_key, full_mask)
        if cached is not None:
            action_idx, value = cached
            return (action_idx, 0.0, value)

        value_fn = self._value_fn()
        candidates = []

        for action in playable_actions:
            action_idx = catanatron_action_to_capstone_index(action)
            if full_mask[action_idx] <= 0.5:
                continue

            child = game.copy()
            child.execute(action)
            immediate_value = value_fn(child, color)
            candidates.append((immediate_value, action_idx, child))

        if not candidates:
            raise ValueError("BeamValuePlacementAgent found no legal placement action")

        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.root_top_k is not None:
            candidates = candidates[: self.root_top_k]

        best_action_idx = None
        best_value = float("-inf")

        for _, action_idx, child in candidates:
            value = self._search(child, color, value_fn, self.search_depth)
            if value > best_value:
                best_action_idx = action_idx
                best_value = value

        self._store_decision(cache_key, best_action_idx, best_value)
        return (best_action_idx, 0.0, best_value)

    def store(self, state, mask, action, log_prob, reward, value, done):
        pass

    def train(self, last_value):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


@register_strategy("rollout_value_selfish")
class SelfishRolloutValuePlacementAgent(RolloutValuePlacementAgent):
    """Complete simulated openings by each player maximizing their own value."""

    def _best_rollout_action(self, game, root_color, value_fn):
        del root_color

        current_color = game.state.current_color()
        best_action = None
        best_value = float("-inf")

        for action in game.playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)
            value = value_fn(game_copy, current_color)
            if value > best_value:
                best_action = action
                best_value = value

        return best_action


@register_strategy("rollout_value_first_roll")
class FirstRolloutValuePlacementAgent(RolloutValuePlacementAgent):
    """Evaluate completed openings by expected value over the next dice roll."""

    def _expected_first_roll_value(self, rollout, root_color, value_fn):
        from catanatron.models.enums import ActionType
        from catanatron.players.tree_search_utils import execute_spectrum

        roll_actions = [
            candidate
            for candidate in rollout.playable_actions
            if candidate.action_type == ActionType.ROLL
        ]
        if not roll_actions:
            return value_fn(rollout, root_color)

        expected_value = 0.0
        for outcome, probability in execute_spectrum(rollout, roll_actions[0]):
            expected_value += probability * value_fn(outcome, root_color)
        return expected_value

    def _completed_opening_value(self, game, action, root_color, value_fn):
        rollout = self._complete_opening(game, action, root_color, value_fn)
        return self._expected_first_roll_value(rollout, root_color, value_fn)


@register_strategy("rollout_value_blend")
class BlendedRolloutValuePlacementAgent(FirstRolloutValuePlacementAgent):
    """Blend completed-opening value with expected value after the first roll."""

    def __init__(self, roll_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.roll_weight = min(1.0, max(0.0, float(roll_weight)))

    def _completed_opening_value(self, game, action, root_color, value_fn):
        rollout = self._complete_opening(game, action, root_color, value_fn)
        base_value = value_fn(rollout, root_color)
        first_roll_value = self._expected_first_roll_value(rollout, root_color, value_fn)
        return (
            (1.0 - self.roll_weight) * base_value
            + self.roll_weight * first_roll_value
        )


@register_strategy("rollout_value_stable")
class StableRolloutValuePlacementAgent(RolloutValuePlacementAgent):
    """Rollout-value agent with deterministic capstone-index tie breaking."""

    TIE_EPSILON = 1e-9

    def _indexed_actions(self, actions):
        from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index

        return sorted(
            (int(catanatron_action_to_capstone_index(action)), action)
            for action in actions
        )

    def _better_value(self, value, best_value, action_idx, best_action_idx, maximizing):
        if best_action_idx is None:
            return True
        if maximizing:
            if value > best_value + self.TIE_EPSILON:
                return True
        elif value < best_value - self.TIE_EPSILON:
            return True

        return (
            abs(value - best_value) <= self.TIE_EPSILON
            and action_idx < best_action_idx
        )

    def _best_rollout_action(self, game, root_color, value_fn):
        maximizing = game.state.current_color() == root_color
        best_action = None
        best_action_idx = None
        best_value = float("-inf") if maximizing else float("inf")

        for action_idx, action in self._indexed_actions(game.playable_actions):
            game_copy = game.copy()
            game_copy.execute(action)
            value = value_fn(game_copy, root_color)
            if self._better_value(
                value,
                best_value,
                action_idx,
                best_action_idx,
                maximizing,
            ):
                best_action = action
                best_action_idx = action_idx
                best_value = value

        return best_action

    def select_action(self, state, mask, **kwargs):
        del state

        game = kwargs.get("game")
        if game is None:
            raise ValueError("StableRolloutValuePlacementAgent requires a `game=` keyword argument")

        playable_actions = kwargs.get("playable_actions") or game.playable_actions
        full_mask = np.asarray(mask, dtype=np.float32)
        color = game.state.current_color()

        from catanatron.gym.envs.action_translator import catanatron_action_to_capstone_index

        cache_key = self._decision_cache_key(
            game,
            playable_actions,
            catanatron_action_to_capstone_index,
        )
        cached = self._cached_decision(cache_key, full_mask)
        if cached is not None:
            action_idx, value = cached
            return (action_idx, 0.0, value)

        value_fn = self._value_fn()
        best_action_idx = None
        best_value = float("-inf")

        for action_idx, action in self._indexed_actions(playable_actions):
            if full_mask[action_idx] <= 0.5:
                continue

            value = self._completed_opening_value(game, action, color, value_fn)
            if self._better_value(
                value,
                best_value,
                action_idx,
                best_action_idx,
                maximizing=True,
            ):
                best_action_idx = action_idx
                best_value = value

        if best_action_idx is None:
            raise ValueError("StableRolloutValuePlacementAgent found no legal placement action")

        self._store_decision(cache_key, best_action_idx, best_value)
        return (best_action_idx, 0.0, best_value)


@register_strategy("rollout_value_stable_first_roll")
class StableFirstRolloutValuePlacementAgent(StableRolloutValuePlacementAgent):
    """Stable rollout search with first-roll expected-value leaf scoring."""

    def _expected_first_roll_value(self, rollout, root_color, value_fn):
        from catanatron.models.enums import ActionType
        from catanatron.players.tree_search_utils import execute_spectrum

        roll_actions = [
            candidate
            for candidate in rollout.playable_actions
            if candidate.action_type == ActionType.ROLL
        ]
        if not roll_actions:
            return value_fn(rollout, root_color)

        expected_value = 0.0
        for outcome, probability in execute_spectrum(rollout, roll_actions[0]):
            expected_value += probability * value_fn(outcome, root_color)
        return expected_value

    def _completed_opening_value(self, game, action, root_color, value_fn):
        rollout = self._complete_opening(game, action, root_color, value_fn)
        return self._expected_first_roll_value(rollout, root_color, value_fn)


@register_strategy("rollout_value_stable_ab_opp")
class StableAlphaBetaOpponentRolloutAgent(StableRolloutValuePlacementAgent):
    """Stable rollout with AlphaBeta-modeled opponent opening replies."""

    def __init__(self, opponent_depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.opponent_depth = max(0, int(opponent_depth))

    def _best_rollout_action(self, game, root_color, value_fn):
        if game.state.current_color() == root_color:
            return super()._best_rollout_action(game, root_color, value_fn)

        from catanatron.players.minimax import AlphaBetaPlayer

        opponent = AlphaBetaPlayer(
            game.state.current_color(),
            depth=self.opponent_depth,
        )
        return opponent.decide(game, game.playable_actions)


# ── learned placement agent ─────────────────────────────────────


class PlacementRolloutBuffer(RolloutBuffer):
    """Placement replay buffer storing prompt-specific local actions."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def store(self, state, mask, action, log_prob, reward, value, done, prompt):
        super().store(state, mask, action, log_prob, reward, value, done)
        self.prompts.append(int(prompt))

    def clear(self):
        """Clear base rollout fields and prompt labels explicitly."""
        super().clear()
        self.prompts = []


@register_strategy("model")
class PlacementAgent:
    """Agent for initial settlement + road placement.

    Shares the same select_action / store / train interface as
    MainPlayAgent so the CapstoneAgent can delegate transparently.

    Public callers still pass the full Capstone observation vector and the
    245-d action mask.  Internally, the agent projects those down to a compact
    opening-only feature space and a prompt-specific local action space.
    """

    MAX_LOCAL_ACTIONS = PlacementModel.EDGE_ACTION_SIZE

    def __init__(
        self,
        obs_size=FEATURE_SPACE_SIZE,
        hidden_size=PLACEMENT_AGENT_HIDDEN_SIZE,
        model_type="mlp",
    ):
        del obs_size  # Kept for backwards compatibility with existing callers.

        self.device = get_device()
        self.hyperparams = PPOHyperparams()
        self.hyperparams.batch_size = 16
        self.model_type = str(model_type).lower()
        if self.model_type not in {"mlp", "gnn"}:
            raise ValueError("model_type must be either 'mlp' or 'gnn'")

        if self.model_type == "gnn":
            self.model = PlacementGNNModel(hidden_dim=hidden_size).to(self.device)
        else:
            self.model = PlacementModel(
                obs_size=COMPACT_PLACEMENT_FEATURE_SIZE,
                hidden_size=hidden_size,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams.lr
        )
        self.buffer = PlacementRolloutBuffer()

    def _forward_model(self, compact_obs_t: torch.Tensor):
        if self.model_type == "gnn":
            batch_size = compact_obs_t.shape[0]
            node_block = compact_obs_t[
                :,
                : NUM_NODES * COMPACT_NODE_FEATURE_SIZE,
            ]
            node_features = node_block.reshape(
                batch_size,
                NUM_NODES,
                COMPACT_NODE_FEATURE_SIZE,
            )
            step_indicators = compact_obs_t[:, -STEP_INDICATOR_SIZE:]
            return self.model(node_features, step_indicators)
        return self.model(compact_obs_t)

    def _compact_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"Expected a 1-d state vector, got shape {state.shape}")
        if state.shape[0] == COMPACT_PLACEMENT_FEATURE_SIZE:
            return state.copy()
        if state.shape[0] == FEATURE_SPACE_SIZE:
            return project_capstone_to_compact_placement(state)
        raise ValueError(
            "Unsupported placement state width "
            f"{state.shape[0]} (expected {FEATURE_SPACE_SIZE} or "
            f"{COMPACT_PLACEMENT_FEATURE_SIZE})"
        )

    def _prepare_public_inputs(self, state, mask):
        full_mask = np.asarray(mask, dtype=np.float32)
        if full_mask.ndim != 1:
            raise ValueError(f"Expected a 1-d action mask, got shape {full_mask.shape}")

        prompt = infer_placement_prompt(full_mask)
        compact_state = self._compact_state(state)
        local_mask = capstone_mask_to_local_mask(full_mask, prompt)
        padded_mask = np.zeros(self.MAX_LOCAL_ACTIONS, dtype=np.float32)
        size = local_action_size(prompt)
        padded_mask[:size] = local_mask
        return compact_state, prompt, padded_mask

    def _prepare_supervised_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"Expected a 2-d observation batch, got shape {obs.shape}")

        if obs.shape[1] == FEATURE_SPACE_SIZE:
            return project_capstone_batch_to_compact_placement(obs)
        if obs.shape[1] == COMPACT_PLACEMENT_FEATURE_SIZE:
            return obs.copy()
        raise ValueError(
            "Unsupported placement observation width "
            f"{obs.shape[1]} (expected {FEATURE_SPACE_SIZE} or "
            f"{COMPACT_PLACEMENT_FEATURE_SIZE})"
        )

    def _prepare_supervised_actions_and_masks(self, masks, actions):
        masks = np.asarray(masks, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        if masks.ndim != 2:
            raise ValueError(f"Expected a 2-d mask batch, got shape {masks.shape}")

        prompts = np.empty(len(masks), dtype=np.int64)
        local_masks = np.zeros((len(masks), self.MAX_LOCAL_ACTIONS), dtype=np.float32)
        local_actions = np.empty(len(masks), dtype=np.int64)

        for idx, full_mask in enumerate(masks):
            prompt = infer_placement_prompt(full_mask)
            prompts[idx] = int(prompt)
            prompt_mask = capstone_mask_to_local_mask(full_mask, prompt)
            prompt_size = local_action_size(prompt)
            local_masks[idx, :prompt_size] = prompt_mask
            local_actions[idx] = capstone_action_to_local(prompt, actions[idx])

        return prompts, local_masks, local_actions

    def _dist_for_single_prompt(
        self,
        settlement_logits,
        road_logits,
        prompt_id,
        masks,
    ):
        """Create a categorical distribution for one prompt-specific batch."""

        prompt = PlacementPrompt(int(prompt_id))
        if prompt == PlacementPrompt.SETTLEMENT:
            logits = settlement_logits
            logits_mask = masks[:, :local_action_size(prompt)]
        else:
            logits = road_logits
            logits_mask = masks[:, :local_action_size(prompt)]

        masked_logits = logits.masked_fill(logits_mask <= 0, -1e9)
        return Categorical(logits=masked_logits)

    def _evaluate_actions(self, states, prompts, masks, actions):
        settlement_logits, road_logits, values = self._forward_model(states)
        log_probs = torch.empty(len(states), device=self.device)
        entropies = torch.empty(len(states), device=self.device)

        for prompt in (PlacementPrompt.SETTLEMENT, PlacementPrompt.ROAD):
            idx = prompts == int(prompt)
            if not torch.any(idx):
                continue

            if prompt == PlacementPrompt.SETTLEMENT:
                logits = settlement_logits[idx]
            else:
                logits = road_logits[idx]

            prompt_masks = masks[idx, :local_action_size(prompt)]
            masked_logits = logits.masked_fill(prompt_masks <= 0, -1e9)
            dist = Categorical(logits=masked_logits)

            log_probs[idx] = dist.log_prob(actions[idx])
            entropies[idx] = dist.entropy()

        return log_probs, entropies, values

    def select_action(self, state, mask, **_kwargs):
        """Sample an action from the policy and return (action, log_prob, value)."""
        compact_state, prompt, local_mask = self._prepare_public_inputs(state, mask)

        state_tensor = torch.from_numpy(compact_state).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(local_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            settlement_logits, road_logits, value = self._forward_model(state_tensor)
            dist = self._dist_for_single_prompt(
                settlement_logits,
                road_logits,
                int(prompt),
                mask_tensor,
            )

        action = dist.sample()
        log_prob = dist.log_prob(action)
        capstone_action = local_action_to_capstone(prompt, action.item())

        return (capstone_action, log_prob.item(), value.item())

    def store(self, state, mask, action, log_prob, reward, value, done):
        compact_state, prompt, local_mask = self._prepare_public_inputs(state, mask)
        local_action = capstone_action_to_local(prompt, action)
        self.buffer.store(
            compact_state,
            local_mask,
            local_action,
            log_prob,
            reward,
            value,
            done,
            int(prompt),
        )

    def compute_advantages(self, last_value):
        advantages = []
        gae = 0

        values = self.buffer.values + [last_value]
        rewards = self.buffer.rewards
        dones = self.buffer.dones

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.hyperparams.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            gae = (
                delta
                + self.hyperparams.gamma
                * self.hyperparams.gae_lambda
                * (1 - dones[t])
                * gae
            )
            advantages.append(gae)

        advantages.reverse()

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.buffer.values).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train(self, last_value):
        """PPO update over collected placement transitions."""
        if len(self.buffer.rewards) == 0:
            return

        advantages, returns = self.compute_advantages(last_value)

        states_np = np.asarray(self.buffer.states, dtype=np.float32)
        masks_np = np.asarray(self.buffer.masks, dtype=np.float32)
        actions_np = np.asarray(self.buffer.actions, dtype=np.int64)
        prompts_np = np.asarray(self.buffer.prompts, dtype=np.int64)
        old_log_probs_np = np.asarray(self.buffer.log_probs, dtype=np.float32)

        states = torch.from_numpy(states_np).to(self.device)
        masks = torch.from_numpy(masks_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        prompts = torch.from_numpy(prompts_np).to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs_np).to(self.device)

        for epoch in range(self.hyperparams.epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.hyperparams.batch_size):
                batch_idx = indices[start : start + self.hyperparams.batch_size]

                b_states = states[batch_idx]
                b_masks = masks[batch_idx]
                b_actions = actions[batch_idx]
                b_prompts = prompts[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                new_log_probs, entropies, values = self._evaluate_actions(
                    b_states,
                    b_prompts,
                    b_masks,
                    b_actions,
                )
                entropy = entropies.mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.hyperparams.clip_eps,
                        1 + self.hyperparams.clip_eps,
                    )
                    * b_advantages
                )
                actor_loss = -torch.where(
                    b_advantages >= 0,
                    torch.min(surr1, surr2),
                    torch.max(surr1, surr2),
                ).mean()

                critic_loss = nn.MSELoss()(values.view(-1), b_returns.view(-1))

                loss = (
                    actor_loss
                    + self.hyperparams.value_coef * critic_loss
                    - self.hyperparams.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.hyperparams.max_grad_norm
                )
                self.optimizer.step()

        self.buffer.clear()

    # ── supervised learning interface ────────────────────────────

    def supervised_train(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        won: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        win_weight: float = 1.0,
        loss_weight: float = 0.1,
    ) -> list:
        """Train on a labelled dataset of placement decisions.

        Args:
            obs:     Either full Capstone observations ``(N, 1259)`` or
                     compact placement observations
                     ``(N, COMPACT_PLACEMENT_FEATURE_SIZE)``.
            masks:   Full Capstone action masks ``(N, 245)``.
            actions: (N,)       action indices that were chosen.
            won:     (N,)       1.0 if the acting player won, else 0.0.
            epochs:  Number of full passes over the dataset.
            batch_size: Mini-batch size.
            lr:      Learning rate (resets the optimizer).
            win_weight:  Sample weight for winning games.
            loss_weight: Sample weight for losing games.

        Returns:
            List of per-epoch mean losses.
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        compact_obs = self._prepare_supervised_obs(obs)
        prompt_ids, local_masks, local_actions = self._prepare_supervised_actions_and_masks(
            masks, actions
        )

        obs_t = torch.as_tensor(compact_obs, dtype=torch.float32).to(self.device)
        mask_t = torch.as_tensor(local_masks, dtype=torch.float32).to(self.device)
        act_t = torch.as_tensor(local_actions, dtype=torch.long).to(self.device)
        prompt_t = torch.as_tensor(prompt_ids, dtype=torch.long).to(self.device)
        weights = torch.where(
            torch.as_tensor(won, dtype=torch.float32) > 0.5,
            win_weight,
            loss_weight,
        ).to(self.device)

        n = len(obs_t)
        epoch_losses = []

        for epoch in range(epochs):
            perm = torch.randperm(n)
            running_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                b_obs = obs_t[idx]
                b_mask = mask_t[idx]
                b_act = act_t[idx]
                b_prompt = prompt_t[idx]
                b_w = weights[idx]

                log_probs, _, _ = self._evaluate_actions(
                    b_obs,
                    b_prompt,
                    b_mask,
                    b_act,
                )
                loss = (-log_probs * b_w).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg = running_loss / max(n_batches, 1)
            epoch_losses.append(avg)

        self.model.eval()
        return epoch_losses

    # ── persistence ──────────────────────────────────────────────

    def load(self, path):
        self.model.load_state_dict(
            torch.load(path, weights_only=True, map_location=self.device)
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)


HeavyPlacementAgent = PlacementAgent
