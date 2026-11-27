from dataclasses import dataclass, fields
from functools import cached_property
import json
import math

from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    RESOURCE_FREQDECK_INDEXES,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    freqdeck_from_listdeck,
)
from catanatron.models.enums import CITY, KNIGHT, ORE, RESOURCES, WHEAT, Action, ActionType
from catanatron.models.map import number_probability
from catanatron.models.player import Player
from catanatron.state_functions import (
    get_actual_victory_points,
    get_largest_army,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_freqdeck,
    get_visible_victory_points,
    player_has_rolled,
    player_key,
)


NEGOTIATION_WEIGHTS = ("offer_threshold", "accept_threshold", "trade_rivalry")


@dataclass(frozen=True)
class Weights:
    diversity: float = 2.6
    wheat_ore: float = 0.30
    port: float = 1.0
    expansion: float = 1.0
    city: float = 1.0
    development: float = 1.0
    trade: float = 1.0
    longest_road: float = 1.0
    offer_threshold: float = 0.24
    accept_threshold: float = 0.14
    trade_rivalry: float = 0.12
    policy_version: int = 1

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {field.name for field in fields(cls)}
        required = names.difference(NEGOTIATION_WEIGHTS, {"policy_version"})
        if not isinstance(data, dict) or not required <= data.keys() <= names:
            raise ValueError("Weights file must contain the core parameters and only supported fields")
        if any(
            type(value) not in (float, int) or not math.isfinite(value) or value <= 0
            for value in data.values()
        ):
            raise ValueError("Each weight must be a finite positive number")
        version = data.get("policy_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported policy version")
        return cls(**data)


def action_order(action):
    return action.action_type.name, str(action.value)


class CatanAgent(Player):
    LABEL = "Catan Agent"

    def __init__(self, color, weights=Weights(), *, negotiate=False):
        super().__init__(color)
        self.weights = weights
        self.negotiate = negotiate
        self.offer_turn = None
        self.offers = set()
        self.replies = {}

    def before(self, game):
        self.offer_turn = None
        self.offers.clear()
        self.replies.clear()

    def step(self, game, action):
        if (
            not self.negotiate
            or action.action_type not in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE)
            or game.state.colors[game.state.current_turn_index] != self.color
        ):
            return
        key = (action.color, tuple(action.value[:10]))
        accepted, total = self.replies.get(key, (0, 0))
        self.replies[key] = (accepted + (action.action_type == ActionType.ACCEPT_TRADE), total + 1)

    def decide(self, game, playable_actions):
        can_offer = self.negotiate and any(
            action.action_type == ActionType.END_TURN for action in playable_actions
        )
        if len(playable_actions) == 1 and not can_offer:
            return playable_actions[0]
        position = Position(game, self.color, self.weights)
        best = max(
            playable_actions,
            key=lambda action: (position.score(action), action_order(action)),
        )
        if self.weights.policy_version >= 2 and any(a.action_type == ActionType.END_TURN for a in playable_actions):
            finish = position.winning_continuation(playable_actions, best)
            if finish is not None:
                return finish
        if can_offer:
            if self.offer_turn != game.state.num_turns:
                self.offer_turn = game.state.num_turns
                self.offers.clear()
            if len(self.offers) < 2:
                offer = self.offer_before_action(position, best)
                if offer is not None:
                    self.offers.add(offer.value)
                    return offer
        return best

    def offer_before_action(self, position, best):
        kind = best.action_type
        if kind == ActionType.END_TURN:
            return self.choose_offer(position, 0.0)
        if kind == ActionType.MARITIME_TRADE:
            # A rejected player offer leaves the bank exchange available.
            factor = 0.4 if self.weights.policy_version >= 2 else 1.0
            return self.choose_offer(position, factor * position.score(best))
        if self.weights.policy_version >= 2 and kind in (ActionType.BUY_DEVELOPMENT_CARD, ActionType.BUILD_ROAD):
            if position.score(best) < 10 and position.build_goals:
                return self.choose_offer(position, 0.0, unlock_build=True)
        return None

    def choose_offer(self, position, alternative_score, *, unlock_build=False):
        best_score = alternative_score
        best = None
        for give_index, held in enumerate(position.hand):
            for quantity in range(1, min(held, 2) + 1):
                giving = tuple(
                    quantity if i == give_index else 0 for i in range(len(RESOURCES))
                )
                for ask_index in range(len(RESOURCES)):
                    if ask_index == give_index or position.other_resources[ask_index] <= 0:
                        continue
                    receiving = tuple(int(i == ask_index) for i in range(len(RESOURCES)))
                    terms = giving + receiving
                    if terms in self.offers:
                        continue
                    if unlock_build:
                        after = [h - g + r for h, g, r in zip(position.hand, giving, receiving)]
                        if not any(all(h >= c for h, c in zip(after, cost)) and
                                   not all(h >= c for h, c in zip(position.hand, cost))
                                   for cost in position.build_goals):
                            continue
                    gain = position.exchange_value(giving, receiving)
                    if gain <= self.weights.offer_threshold:
                        continue
                    expected_gain = 0.0
                    for partner in position.state.colors:
                        if partner == self.color:
                            continue
                        accepted, total = self.replies.get((partner, terms), (0, 0))
                        prior = min(0.65, 0.20 + 0.06 * position.other_resources[ask_index])
                        likelihood = (2 * prior + accepted) / (2 + total)
                        surplus = gain - self.weights.trade_rivalry * position.threat(partner)
                        expected_gain = max(expected_gain, likelihood * surplus)
                    score = self.weights.trade * expected_gain - self.weights.offer_threshold
                    if score > best_score:
                        best_score = score
                        best = Action(self.color, ActionType.OFFER_TRADE, terms)
        return best


class Position:
    def __init__(self, game, color, weights):
        self.state = game.state
        self.board = game.state.board
        self.color = color
        self.weights = weights
        self.target = game.vps_to_win
        self.key = player_key(self.state, color)

    @cached_property
    def production(self):
        totals = dict.fromkeys(RESOURCES, 0.0)
        for node, (owner, building) in self.board.buildings.items():
            if owner == self.color:
                for resource, probability in self.board.map.node_production[node].items():
                    totals[resource] += probability * (72 if building == CITY else 36)
        return totals

    @cached_property
    def hand(self):
        return get_player_freqdeck(self.state, self.color)

    @cached_property
    def other_resources(self):
        return [19 - bank - own for bank, own in zip(self.state.resource_freqdeck, self.hand)]

    def threat(self, partner):
        return max(0, get_visible_victory_points(self.state, partner) - 6)

    @cached_property
    def points(self):
        return get_actual_victory_points(self.state, self.color)

    def winning_continuation(self, actions, best):
        """Find a guaranteed final point using this turn's public legal moves."""
        if self.points != self.target - 1 or self.score(best) >= 1000:
            return None
        villages = self.state.player_state[f"{self.key}_SETTLEMENTS_AVAILABLE"]
        goals = self.build_goals
        for action in actions:
            if action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
                gain = freqdeck_from_listdeck(action.value)
                if any(all(h + g >= c for h, g, c in zip(self.hand, gain, cost)) for cost in goals):
                    return action
        if villages and all(h >= r + s for h, r, s in zip(self.hand, ROAD_COST_FREQDECK, SETTLEMENT_COST_FREQDECK)):
            for action in actions:
                if action.action_type == ActionType.BUILD_ROAD and any(v in self.board.board_buildable_ids for v in action.value):
                    return action

        trades = [a for a in actions if a.action_type == ActionType.MARITIME_TRADE]
        if not trades:
            return None
        bank = getattr(self.state, "resource_freqdeck", None)
        ratios = [2 if r in self.ports else 3 if None in self.ports else 4 for r in RESOURCES]
        for cost in goals:
            short = [max(0, c - h) for c, h in zip(cost, self.hand)]
            extra = [max(0, h - c) for c, h in zip(cost, self.hand)]
            if (bank is None and sum(short) != 1) or (bank is not None and any(s > b for s, b in zip(short, bank))):
                continue
            if sum(short) > sum(e // r for e, r in zip(extra, ratios)):
                continue
            for action in trades:
                give = RESOURCE_FREQDECK_INDEXES[action.value[0]]
                want = RESOURCE_FREQDECK_INDEXES[action.value[-1]]
                price = sum(resource is not None for resource in action.value[:-1])
                if short[want] and price == ratios[give] and extra[give] >= price:
                    return action
        return None

    @cached_property
    def build_goals(self):
        goals = []
        if self.state.player_state[f"{self.key}_CITIES_AVAILABLE"] and any(
            owner == self.color and kind != CITY for owner, kind in self.board.buildings.values()
        ):
            goals.append(CITY_COST_FREQDECK)
        if self.state.player_state[f"{self.key}_SETTLEMENTS_AVAILABLE"] and self.connected_sites:
            goals.append(SETTLEMENT_COST_FREQDECK)
        return goals

    @cached_property
    def connected_sites(self):
        return set(self.board.buildable_node_ids(self.color))

    @cached_property
    def ports(self):
        return self.board.get_player_port_resources(self.color)

    def site_value(self, node, city=False):
        pips = self.board.map.node_production[node]
        production = 0.0
        for resource, probability in pips.items():
            marginal = 36 * probability
            production += marginal / (1 + 0.035 * self.production[resource])
            if resource in (WHEAT, ORE):
                production += self.weights.wheat_ore * marginal
            if self.production[resource] == 0:
                production += self.weights.diversity * min(marginal / 3, 1)
        if city:
            return production
        for resource, nodes in self.board.map.port_nodes.items():
            if node in nodes and resource not in self.ports:
                port_value = (
                    1.8 if resource is None
                    else min(3.5, (self.production[resource] + 36 * pips[resource]) / 2)
                )
                production += self.weights.port * port_value
        return production

    def road_score(self, edge):
        # Look for new sites one or two roads away from this edge.
        sites = self.board.board_buildable_ids - self.connected_sites
        best = 0.0
        for node in edge:
            if node in sites:
                best = max(best, self.site_value(node) / 6 + 1.4)
            if self.board.is_enemy_node(node, self.color):
                continue
            for neighbor in self.board.buildable_subgraph[node]:
                if (
                    neighbor in sites and neighbor not in edge
                    and (node, neighbor) not in self.board.roads
                ):
                    best = max(best, self.site_value(neighbor) / 10 + 0.2)

        if self.state.is_initial_build_phase:
            return best

        settlements_left = self.state.player_state[f"{self.key}_SETTLEMENTS_AVAILABLE"]
        score = -2.0
        if settlements_left:
            score += best * self.weights.expansion / (1 + len(self.connected_sites))

        # Check longest-road gains when they can plausibly decide the game.
        if self.points >= 6 or self.state.is_road_building or sum(self.hand) >= 8:
            previous_length = get_longest_road_length(self.state, self.color)
            after = self.board.copy()
            old_owner, new_owner, lengths = after.build_road(self.color, edge)
            if old_owner != self.color and new_owner == self.color:
                if self.points + 2 >= self.target:
                    return 1000.0
                score += 4.0 * self.weights.longest_road
            elif old_owner != self.color:
                gain = lengths[self.color] - previous_length
                gap = max(0, max(5, self.board.road_length + 1) - lengths[self.color])
                score += self.weights.longest_road * gain * 2.6 / (1 + gap)
        return score + (0.45 if sum(self.hand) >= 8 else 0.0)

    def robber_score(self, coordinate, victim):
        tile = self.board.map.land_tiles[coordinate]
        blocked = 0.0
        buildings = (self.board.buildings.get(node) for node in tile.nodes.values())
        for building in buildings:
            if building is None:
                continue
            owner, kind = building
            threat = 1 + 0.16 * get_visible_victory_points(self.state, owner)
            blocked += (-3 if owner == self.color else threat) * (2 if kind == CITY else 1)
        steal = 0 if victim is None else 2 + 0.22 * get_visible_victory_points(self.state, victim)
        return 36 * number_probability(tile.number) * blocked + steal

    @cached_property
    def hand_plans(self):
        plans = [(DEVELOPMENT_CARD_COST_FREQDECK, 0.8 * self.weights.development)]
        if self.state.player_state[f"{self.key}_CITIES_AVAILABLE"]:
            plans.append((CITY_COST_FREQDECK, 1.3 * self.weights.city))
        if self.state.player_state[f"{self.key}_SETTLEMENTS_AVAILABLE"]:
            plans.append((SETTLEMENT_COST_FREQDECK, 1.3 if self.connected_sites else 0.8))
        return plans

    def hand_value(self, hand):
        progress = []
        for cost, multiplier in self.hand_plans:
            missing = sum(max(0, need - have) for need, have in zip(cost, hand))
            progress.append(multiplier * (1 + (missing == 0)) / (1 + missing))
        flexibility = sum(
            min(count, 5) / (1 + self.production[resource] / 8)
            for resource, count in zip(RESOURCES, hand)
        )
        return max(progress) + 0.25 * sum(progress) + 0.10 * flexibility

    @cached_property
    def current_hand_value(self):
        return self.hand_value(self.hand)

    def exchange_value(self, giving, receiving):
        after = [count - give + receive for count, give, receive in zip(self.hand, giving, receiving)]
        return self.hand_value(after) - self.current_hand_value

    def score(self, action):
        kind, value = action.action_type, action.value
        match kind:
            case ActionType.BUILD_SETTLEMENT:
                if self.points + 1 >= self.target:
                    return 1000.0
                return 5.5 + 0.3 * self.site_value(value)
            case ActionType.BUILD_CITY:
                if self.points + 1 >= self.target:
                    return 1000.0
                return 6.0 * self.weights.city + 0.3 * self.site_value(value, city=True)
            case ActionType.BUILD_ROAD:
                return self.road_score(value)
            case ActionType.MOVE_ROBBER:
                return self.robber_score(*value)
            case ActionType.DISCARD_RESOURCE:
                hand = self.hand.copy()
                hand[RESOURCE_FREQDECK_INDEXES[value]] -= 1
                return self.hand_value(hand)
            case ActionType.BUY_DEVELOPMENT_CARD:
                ore = self.hand[RESOURCE_FREQDECK_INDEXES[ORE]]
                wheat = self.hand[RESOURCE_FREQDECK_INDEXES[WHEAT]]
                return 2.6 * self.weights.development + 0.25 * self.points - 0.16 * min(ore, wheat)
            case ActionType.PLAY_KNIGHT_CARD:
                owner, army = get_largest_army(self.state)
                played = get_played_dev_cards(self.state, self.color, KNIGHT)
                if owner != self.color and played + 1 >= max(3, (army or 0) + 1):
                    return 1000.0 if self.points + 2 >= self.target else 10.0
                blocked = self.board.map.land_tiles[self.board.robber_coordinate]
                if any(
                    self.board.is_friendly_node(node, self.color)
                    for node in blocked.nodes.values()
                ):
                    return 5.0
                return 1.2 if owner != self.color else -0.4
            case ActionType.PLAY_YEAR_OF_PLENTY:
                if not player_has_rolled(self.state, self.color):
                    return -1.0
                after = [
                    count + extra
                    for count, extra in zip(self.hand, freqdeck_from_listdeck(value))
                ]
                return 2.0 * (self.hand_value(after) - self.hand_value(self.hand)) - 0.25
            case ActionType.PLAY_ROAD_BUILDING:
                return 4.0 if player_has_rolled(self.state, self.color) else -1.0
            case ActionType.PLAY_MONOPOLY:
                if not player_has_rolled(self.state, self.color):
                    return -1.0
                index = RESOURCE_FREQDECK_INDEXES[value]
                # Infer demand from the public bank and our own hand only.
                held_by_others = 19 - self.state.resource_freqdeck[index] - self.hand[index]
                after = self.hand.copy()
                after[index] += held_by_others
                hand_gain = self.hand_value(after) - self.hand_value(self.hand)
                return hand_gain + 0.48 * held_by_others - 1.65
            case ActionType.MARITIME_TRADE:
                giving = freqdeck_from_listdeck(
                    resource for resource in value[:-1] if resource is not None
                )
                receiving = freqdeck_from_listdeck(value[-1:])
                return self.weights.trade * self.exchange_value(giving, receiving) - 0.20
            case ActionType.ACCEPT_TRADE:
                proposer = self.state.colors[self.state.current_turn_index]
                return (
                    self.exchange_value(value[5:10], value[:5])
                    - self.weights.accept_threshold
                    - self.weights.trade_rivalry * self.threat(proposer)
                )
            case ActionType.CONFIRM_TRADE:
                return (
                    self.exchange_value(value[:5], value[5:10])
                    - self.weights.trade_rivalry * self.threat(value[10])
                )
            case (
                ActionType.ROLL | ActionType.END_TURN
                | ActionType.REJECT_TRADE | ActionType.CANCEL_TRADE
            ):
                return 0.0
            case _:
                raise ValueError(f"Unsupported action {kind}")
