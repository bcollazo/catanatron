"""
The engine-side adapter that feeds fair agents Observations.

The game loop always hands Player.decide a perfect-information Game.
PerspectivePlayer converts that Game into an Observation for the wrapped
agent on every decision, so the agent never perceives more than a real
player could observe.
"""

from catanatron.features import create_sample
from catanatron.models.enums import (
    Action,
    ActionRecord,
    ActionType,
    DEVELOPMENT_CARDS,
    RESOURCES,
)
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS
from catanatron.models.inventory import Inventory
from catanatron.models.observation import Observation
from catanatron.models.player import Player
from catanatron.models.public_state import (
    PublicBoard,
    PublicMap,
    PublicPlayer,
    PublicState,
)
from catanatron.models.tiles import LandTile
from catanatron.models.trade import PendingTrades, TradeOffer
from catanatron.state_functions import (
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)


def _sanitize_record(record, observer_color):
    """Returns a copy of the record with hidden identities redacted.

    Full detail is retained for the observer's own records. For opponent
    records: development-card purchases are redacted (both channels), and
    stolen-card identities are redacted unless the observer was the victim.
    Discards are public per tournament convention.
    """
    action = record.action
    if action.color == observer_color:
        return record

    action_type = action.action_type
    if action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return ActionRecord(Action(action.color, action_type, None), None)
    elif action_type == ActionType.MOVE_ROBBER:
        robbed_color = action.value[1] if action.value is not None else None
        if robbed_color == observer_color:
            return record
        return ActionRecord(action, None)
    return record


def _sanitize_history(game, observer_color):
    """Sanitizes the full-truth action log for the given observer color."""
    return [
        _sanitize_record(record, observer_color) for record in game.state.action_records
    ]


def _build_public_map(board):
    """Projects the engine's static CatanMap into pure-data form.

    The map never changes during a game, so every decision snapshots the same
    terrain; it is still projected here so ``Observation`` stays free of any
    engine reference and works unchanged in site/adapter mode.
    """
    catan_map = board.map
    tiles = {}
    for tile in catan_map.tiles_by_id.values():
        tiles[tile.id] = (tile.resource, tile.number)
    tile_coordinates = {
        tile.id: coordinate
        for coordinate, tile in catan_map.tiles.items()
        if isinstance(tile, LandTile)
    }
    ports = {}
    for port in catan_map.ports_by_id.values():
        (a_ref, b_ref) = PORT_DIRECTION_TO_NODEREFS[port.direction]
        ports[port.id] = (port.resource, (port.nodes[a_ref], port.nodes[b_ref]))
    adjacent_tiles = {
        node_id: tuple(t.id for t in tiles_list)
        for node_id, tiles_list in catan_map.adjacent_tiles.items()
    }
    return PublicMap(
        tiles=tiles,
        tile_coordinates=tile_coordinates,
        ports=ports,
        adjacent_tiles=adjacent_tiles,
        land_nodes=frozenset(catan_map.land_nodes),
    )


def _build_public_state(game):
    """Projects the engine state down to a pure-data snapshot of public facts.

    Everything here is knowable with certainty by every player: the buildings
    map, roads, robber, longest-road holder, per-color public counts, and the
    static board layout. It is keyed absolutely (node/edge/tile ids and
    Colors), unlike ``features`` which is keyed relative to the observing
    color. Opponent hand identities and actual victory points are never
    included.
    """
    state = game.state
    board = state.board

    roads = {}
    for edge, color in board.roads.items():
        node_a, node_b = edge
        if node_a < node_b:
            roads[(node_a, node_b)] = color

    players = {}
    for color in state.colors:
        key = player_key(state, color)
        player_state = state.player_state
        played = {
            f"played_{card.lower()}": player_state[f"{key}_PLAYED_{card}"]
            for card in DEVELOPMENT_CARDS
        }
        players[color] = PublicPlayer(
            public_vps=player_state[f"{key}_VICTORY_POINTS"],
            has_army=player_state[f"{key}_HAS_ARMY"],
            has_road=player_state[f"{key}_HAS_ROAD"],
            longest_road_length=player_state[f"{key}_LONGEST_ROAD_LENGTH"],
            roads_left=player_state[f"{key}_ROADS_AVAILABLE"],
            settlements_left=player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
            cities_left=player_state[f"{key}_CITIES_AVAILABLE"],
            has_rolled=player_state[f"{key}_HAS_ROLLED"],
            hand_resource_count=player_num_resource_cards(state, color),
            hand_dev_count=player_num_dev_cards(state, color),
            **played,
        )

    return PublicState(
        board=PublicBoard(
            buildings=dict(board.buildings),
            roads=roads,
            robber_tile_id=board.map.tiles[board.robber_coordinate].id,
            longest_road_color=board.road_color,
            longest_road_length=board.road_length,
            map=_build_public_map(board),
        ),
        players=players,
    )


def _build_pending_trades(game):
    """Projects the engine's active trade state into typed TradeOffers.

    The engine holds a single ``current_trade`` slot, so today this yields at
    most one offer; the tuple shape future-proofs site-mode adapters that allow
    competing offers. Trades are fully public, so nothing here is redacted.
    """
    state = game.state
    if not state.is_resolving_trade:
        return PendingTrades()

    last_offer = next(
        r.action
        for r in reversed(state.action_records)
        if r.action.action_type == ActionType.OFFER_TRADE
    )
    offered = dict(zip(RESOURCES, state.current_trade[:5]))
    asking = dict(zip(RESOURCES, state.current_trade[5:10]))
    return PendingTrades(
        (
            TradeOffer(
                offerer=last_offer.color,
                offered=offered,
                asking=asking,
                acceptees=dict(zip(state.colors, state.acceptees)),
            ),
        )
    )


def _build_inventory(game, color):
    """Projects the observing color's private hand into a typed Inventory.

    The observer's own exact resource counts and dev-card identities are
    knowable only to that color, so this is computed for the observer only.
    """
    key = player_key(game.state, color)
    player_state = game.state.player_state
    hand = {r.lower(): player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES}
    hand.update(
        {c.lower(): player_state[f"{key}_{c}_IN_HAND"] for c in DEVELOPMENT_CARDS}
    )
    return Inventory(
        **hand,
        actual_vps=player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
        has_played_development_card=player_state[
            f"{key}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"
        ],
    )


class PerspectivePlayer(Player):
    """Player that adapts an ObservationAgent into the perfect-info game loop.

    On every decision the given game is projected down to the agent's color
    as an Observation (features, sanitized history, trade state, and a
    structured public-state snapshot) and the agent's decide_observation is
    called. The agent never receives the Game.

    Args:
        agent (ObservationAgent): the fair agent to drive.
        is_bot (bool): whether this player is a bot. Defaults to True.
    """

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
            pending_trades=_build_pending_trades(game),
            public_state=_build_public_state(game),
            inventory=_build_inventory(game, color),
        )
        return self.agent.decide_observation(observation, playable_actions)

    def reset_state(self):
        self.agent.reset_state()
