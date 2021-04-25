import random
from collections import defaultdict

from catanatron.models.board import Board
from catanatron.models.enums import DevelopmentCard, Resource, BuildingType
from catanatron.models.decks import DevelopmentDeck, ResourceDeck
from catanatron.models.actions import Action, ActionPrompt, ActionType
from catanatron.algorithms import longest_road, largest_army


class State:
    """Small container object to group dynamic variables in state"""

    def __init__(self, players, catan_map, initialize=True):
        if initialize:
            self.players = random.sample(players, len(players))
            self.players_by_color = {p.color: p for p in players}
            self.board = Board(catan_map)
            self.actions = []  # log of all action taken by players
            self.resource_deck = ResourceDeck.starting_bank()
            self.development_deck = DevelopmentDeck.starting_bank()
            self.tick_queue = initialize_tick_queue(self.players)
            self.current_player_index = 0
            self.num_turns = 0  # num_completed_turns
            self.road_color = None
            self.army_color = None

            # To be set by Game
            self.current_prompt = None
            self.playable_actions = None

    def current_player(self):
        return self.players[self.current_player_index]


def initialize_tick_queue(players):
    """First player goes, settlement and road, ..."""
    tick_queue = []
    for seating in range(len(players)):
        tick_queue.append((seating, ActionPrompt.BUILD_FIRST_SETTLEMENT))
        tick_queue.append((seating, ActionPrompt.BUILD_INITIAL_ROAD))
    for seating in range(len(players) - 1, -1, -1):
        tick_queue.append((seating, ActionPrompt.BUILD_SECOND_SETTLEMENT))
        tick_queue.append((seating, ActionPrompt.BUILD_INITIAL_ROAD))
    tick_queue.append((0, ActionPrompt.ROLL))
    return tick_queue


def roll_dice():
    return (random.randint(1, 6), random.randint(1, 6))


def yield_resources(board, resource_deck, number):
    """
    Returns:
        (payouts, depleted): tuple where:
        payouts: dictionary of "resource_deck" keyed by player
                e.g. {Color.RED: ResourceDeck({Resource.WEAT: 3})}
            depleted: list of resources that couldn't yield
    """
    intented_payout = defaultdict(lambda: defaultdict(int))
    resource_totals = defaultdict(int)
    for coordinate, tile in board.map.resource_tiles:
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for _, node_id in tile.nodes.items():
            building = board.buildings.get(node_id, None)
            if building is None:
                continue
            elif building[1] == BuildingType.SETTLEMENT:
                intented_payout[building[0]][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building[1] == BuildingType.CITY:
                intented_payout[building[0]][tile.resource] += 2
                resource_totals[tile.resource] += 2

    # for each resource, check enough in deck to yield.
    depleted = []
    for resource in Resource:
        total = resource_totals[resource]
        if not resource_deck.can_draw(total, resource):
            depleted.append(resource)

    # build final data ResourceDeck structure
    payout = {}
    for player, player_payout in intented_payout.items():
        payout[player] = ResourceDeck()

        for resource, count in player_payout.items():
            if resource not in depleted:
                payout[player].replenish(count, resource)

    return payout, depleted


def apply_action(state, action):
    if action.action_type == ActionType.END_TURN:
        next_player_index = (state.current_player_index + 1) % len(state.players)
        state.current_player_index = next_player_index
        state.players[next_player_index].clean_turn_state()
        state.tick_queue.append((next_player_index, ActionPrompt.ROLL))
        state.num_turns += 1
    elif action.action_type == ActionType.BUILD_FIRST_SETTLEMENT:
        player, node_id = state.players_by_color[action.color], action.value
        state.board.build_settlement(player.color, node_id, True)
        player.build_settlement(node_id, True)
    elif action.action_type == ActionType.BUILD_SECOND_SETTLEMENT:
        player, node_id = state.players_by_color[action.color], action.value
        state.board.build_settlement(player.color, node_id, True)
        player.build_settlement(node_id, True)
        # yield resources of second settlement
        for tile in state.board.map.adjacent_tiles[node_id]:
            if tile.resource != None:
                state.resource_deck.draw(1, tile.resource)
                player.resource_deck.replenish(1, tile.resource)
    elif action.action_type == ActionType.BUILD_SETTLEMENT:
        player, node_id = state.players_by_color[action.color], action.value
        state.board.build_settlement(player.color, node_id, False)
        player.build_settlement(node_id, False)
        state.resource_deck += ResourceDeck.settlement_cost()  # replenish bank
        state.road_color = longest_road(state.board, state.players, state.actions)[0]
    elif action.action_type == ActionType.BUILD_INITIAL_ROAD:
        player, edge = state.players_by_color[action.color], action.value
        state.board.build_road(player.color, edge)
        player.build_road(edge, True)
    elif action.action_type == ActionType.BUILD_ROAD:
        player, edge = state.players_by_color[action.color], action.value
        state.board.build_road(player.color, edge)
        player.build_road(edge, False)
        state.resource_deck += ResourceDeck.road_cost()  # replenish bank
        state.road_color = longest_road(state.board, state.players, state.actions)[0]
    elif action.action_type == ActionType.BUILD_CITY:
        player, node_id = state.players_by_color[action.color], action.value
        state.board.build_city(player.color, node_id)
        player.build_city(node_id)
        state.resource_deck += ResourceDeck.city_cost()  # replenish bank
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        player = state.players_by_color[action.color]
        if state.development_deck.num_cards() == 0:
            raise ValueError("No more development cards")
        if not player.resource_deck.includes(ResourceDeck.development_card_cost()):
            raise ValueError("No money to buy development card")

        if action.value is None:
            card = state.development_deck.random_draw()
        else:
            card = action.value
            state.development_deck.draw(1, card)

        player.development_deck.replenish(1, card)
        player.resource_deck -= ResourceDeck.development_card_cost()
        state.resource_deck += ResourceDeck.development_card_cost()

        action = Action(action.color, action.action_type, card)
    elif action.action_type == ActionType.ROLL:
        player = state.players_by_color[action.color]
        player.has_rolled = True
        dices = action.value or roll_dice()
        number = dices[0] + dices[1]

        if number == 7:
            seatings_to_discard = [
                seating
                for seating, player in enumerate(state.players)
                if player.resource_deck.num_cards() > 7
            ]
            state.tick_queue.extend(
                [(seating, ActionPrompt.DISCARD) for seating in seatings_to_discard]
            )
            state.tick_queue.append(
                (state.current_player_index, ActionPrompt.MOVE_ROBBER)
            )
        else:
            payout, _ = yield_resources(state.board, state.resource_deck, number)
            for color, resource_deck in payout.items():
                player = state.players_by_color[color]

                # Atomically add to player's hand and remove from bank
                player.resource_deck += resource_deck
                state.resource_deck -= resource_deck

        action = Action(action.color, action.action_type, dices)
        state.tick_queue.append((state.current_player_index, ActionPrompt.PLAY_TURN))
    elif action.action_type == ActionType.DISCARD:
        player = state.players_by_color[action.color]
        hand = player.resource_deck.to_array()
        num_to_discard = len(hand) // 2
        if action.value is None:
            # TODO: Forcefully discard randomly so that decision tree doesnt explode in possibilities.
            discarded = random.sample(hand, k=num_to_discard)
        else:
            discarded = action.value  # for replay functionality
        to_discard = ResourceDeck.from_array(discarded)

        player.resource_deck -= to_discard
        state.resource_deck += to_discard
        action = Action(action.color, action.action_type, discarded)
    elif action.action_type == ActionType.MOVE_ROBBER:
        player = state.players_by_color[action.color]
        (coordinate, robbed_color, robbed_resource) = action.value
        state.board.robber_coordinate = coordinate
        if robbed_color is not None:
            player_to_steal_from = state.players_by_color[robbed_color]
            enemy_cards = player_to_steal_from.resource_deck.num_cards()
            if robbed_resource is None:
                resource = player_to_steal_from.resource_deck.random_draw()
                action = Action(
                    action.color,
                    action.action_type,
                    (coordinate, robbed_color, resource),
                )
            else:  # for replay functionality
                resource = robbed_resource
                player_to_steal_from.resource_deck.draw(1, resource)
            player.resource_deck.replenish(1, resource)
    elif action.action_type == ActionType.PLAY_KNIGHT_CARD:
        player = state.players_by_color[action.color]
        if not player.can_play_knight():
            raise ValueError("Player cant play knight card now")
        (coordinate, robbed_color, robbed_resource) = action.value
        state.board.robber_coordinate = coordinate
        if robbed_color is not None:
            player_to_steal_from = state.players_by_color[robbed_color]
            enemy_cards = player_to_steal_from.resource_deck.num_cards()
            if robbed_resource is None:
                resource = player_to_steal_from.resource_deck.random_draw()
                action = Action(
                    action.color,
                    action.action_type,
                    (coordinate, robbed_color, resource),
                )
            else:  # for replay functionality
                resource = robbed_resource
                player_to_steal_from.resource_deck.draw(1, resource)
            player.resource_deck.replenish(1, resource)
        player.mark_played_dev_card(DevelopmentCard.KNIGHT)
        state.army_color = largest_army(state.players, state.actions)[0]
    elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        player = state.players_by_color[action.color]
        cards_selected = ResourceDeck.from_array(action.value)
        if not player.can_play_year_of_plenty():
            raise ValueError("Player cant play year of plenty now")
        if not state.resource_deck.includes(cards_selected):
            raise ValueError("Not enough resources of this type (these types?) in bank")
        player.resource_deck += cards_selected
        state.resource_deck -= cards_selected
        player.mark_played_dev_card(DevelopmentCard.YEAR_OF_PLENTY)
    elif action.action_type == ActionType.PLAY_MONOPOLY:
        player, mono_resource = (
            state.players_by_color[action.color],
            action.value,
        )
        cards_stolen = ResourceDeck()
        if not player.can_play_monopoly():
            raise ValueError("Player cant play monopoly now")
        total_enemy_cards = 0
        for p in state.players:
            if not p.color == action.color:
                number_of_cards_to_steal = p.resource_deck.count(mono_resource)
                cards_stolen.replenish(number_of_cards_to_steal, mono_resource)
                p.resource_deck.draw(number_of_cards_to_steal, mono_resource)
                total_enemy_cards += p.resource_deck.num_cards()
        player.resource_deck += cards_stolen
        player.mark_played_dev_card(DevelopmentCard.MONOPOLY)
    elif action.action_type == ActionType.PLAY_ROAD_BUILDING:
        player, (first_edge, second_edge) = (
            state.players_by_color[action.color],
            action.value,
        )
        if not player.can_play_road_building():
            raise ValueError("Player cant play road building now")

        state.board.build_road(player.color, first_edge)
        state.board.build_road(player.color, second_edge)
        player.build_road(first_edge, True)
        player.build_road(second_edge, True)
        player.mark_played_dev_card(DevelopmentCard.ROAD_BUILDING)
        state.road_color = longest_road(state.board, state.players, state.actions)[0]
    elif action.action_type == ActionType.MARITIME_TRADE:
        player, trade_offer = (state.players_by_color[action.color], action.value)
        offering = ResourceDeck.from_array(
            filter(lambda r: r is not None, trade_offer[:-1])
        )
        asking = ResourceDeck.from_array(trade_offer[-1:])
        if not player.resource_deck.includes(offering):
            raise ValueError("Trying to trade without money")
        if not state.resource_deck.includes(asking):
            raise ValueError("Bank doenst have those cards")
        player.resource_deck -= offering
        state.resource_deck += offering
        player.resource_deck += asking
        state.resource_deck -= asking
    else:
        raise RuntimeError("Unknown ActionType " + str(action.action_type))
    return action
