import os
import json
import pickle

pickle_game_states_dir = "game_states_pickle"

# retrieve uuid of all games
def get_games_info():
    # list all folders in game_states_pickle
    games_uuid = [name for name in os.listdir(pickle_game_states_dir) if os.path.isdir(os.path.join(pickle_game_states_dir, name))]
    # count number of folders in game_states_pickle
    game_count = len(games_uuid)

    return game_count, games_uuid

# serialize and store game state using pickle
def serialize_game_state(game, state_index):
    pickle_game_dir = f"{pickle_game_states_dir}/{game.id}"
    
    # create pickle directory if it doesn't exist
    if not os.path.exists(pickle_game_dir):
        os.makedirs(pickle_game_dir)

    # save pickle data to file
    with open(f"{pickle_game_dir}/{state_index}.pickle", "wb") as f:
        pickle.dump(game, f)


def save_game_metadata(game):
    pickle_game_dir = f"{pickle_game_states_dir}/{game.id}"

    # create pickle directory if it doesn't exist
    if not os.path.exists(pickle_game_dir):
        os.makedirs(pickle_game_dir)

    # save game metadata as a json file
    with open(f"{pickle_game_dir}/metadata.json", "w") as f:
        metadata = {
            "id": game.id,
            "players": [str(player.value) for player in game.state.colors],
            "winner": str(game.winning_color()),
            "game_states_count": game.state_index + 1
        }
        json.dump(metadata, f)


# retrieve all game states of a game
def get_game_metadata(game_id):
    pickle_game_dir = f"{pickle_game_states_dir}/{game_id}"
    
    # check if game folder exists
    if not os.path.exists(f"{pickle_game_dir}"):
        return None

    # read metadata.json file
    with open(f"{pickle_game_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
        return metadata


# load game state from pickle file
def load_game_state(game_id, state_index):
    pickle_game_dir = f"{pickle_game_states_dir}/{game_id}"

    # check if game folder exists
    if not os.path.exists(f"{pickle_game_dir}"):
        return None

    if state_index is None:
        metadata = get_game_metadata(game_id)
        if metadata is None:
            return None
        game_states = metadata["game_states_count"]
        state_index = game_states - 1

    # check if file exists
    if not os.path.exists(f"{pickle_game_dir}/{state_index}.pickle"):
        return None

    with open(f"{pickle_game_dir}/{state_index}.pickle", "rb") as f:
        game = pickle.load(f)
    
    return game