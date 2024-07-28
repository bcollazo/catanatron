import os
import json
import pickle


# save game metadata as a json file in the pickle directory
def save_game_pickle_metadata(game, pickle_game_states_dir):
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
            "game_states_count": len(game.state.actions),
        }
        json.dump(metadata, f)


# serialize and store game state using pickle
def save_game_state_to_pickle(game, state_index, pickle_game_states_dir):
    pickle_game_dir = f"{pickle_game_states_dir}/{game.id}"

    # create pickle directory if it doesn't exist
    if not os.path.exists(pickle_game_dir):
        os.makedirs(pickle_game_dir)

    # save pickle data to file
    with open(f"{pickle_game_dir}/{state_index}.pickle", "wb") as f:
        pickle.dump(game, f)

    # update metadata
    save_game_pickle_metadata(game, pickle_game_states_dir)
