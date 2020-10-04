import os
import json
import pickle

import psycopg2

from catanatron.json import GameEncoder

CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS game_states (
        id              SERIAL PRIMARY KEY,
        game_id         VARCHAR(36) NOT NULL,
        action_index    INTEGER, 
        state           JSON, 
        pickle_data     BYTEA,

        UNIQUE(game_id, action_index)
    )
    """
INSERT_STATE_QUERY = """
    INSERT INTO game_states (game_id, action_index, state, pickle_data) VALUES (%s, %s, %s, %s)
    """
UPSERT_STATE_QUERY = """
    INSERT INTO game_states (game_id, action_index, state, pickle_data) VALUES (%s, %s, %s, %s)
        ON CONFLICT(game_id, action_index) 
        DO UPDATE SET 
            state=excluded.state,
            pickle_data=excluded.pickle_data;
    """
SELECT_GAME_QUERY = """
    SELECT pickle_data FROM game_states 
    WHERE game_id = %s ORDER BY action_index DESC LIMIT 1
"""
SELECT_GAME_IDS_QUERY = """SELECT DISTINCT game_id FROM game_states"""
SELECT_STATES_QUERY = """SELECT * FROM game_states WHERE game_id = %s"""

connection = psycopg2.connect(
    user="catanatron",
    password="victorypoint",
    host="127.0.0.1",
    port="5432",
    database="catanatron_db",
)
cursor = connection.cursor()

cursor.execute(CREATE_TABLE_QUERY)
connection.commit()


def save_game_state(game):
    state = json.dumps(game, cls=GameEncoder)
    pickle_data = pickle.dumps(game, pickle.HIGHEST_PROTOCOL)
    cursor.execute(
        INSERT_STATE_QUERY,
        (game.id, len(game.actions), state, pickle_data),
    )
    connection.commit()


def get_last_game_state(game_id):
    cursor.execute(SELECT_GAME_QUERY, (game_id,))
    row = cursor.fetchone()
    if row is None:
        return None

    pickle_data = row[0]
    game = pickle.loads(pickle_data)
    return game


# TODO: Filter by winners
def get_finished_games_ids():
    cursor.execute(SELECT_GAME_IDS_QUERY)
    row = cursor.fetchone()
    while row is not None:
        yield row[0]
        row = cursor.fetchone()


def get_game_states(game_id):
    cursor.execute(SELECT_STATES_QUERY, (game_id,))
    row = cursor.fetchone()
    while row is not None:
        pickle_data = row[4]
        game = pickle.loads(pickle_data)
        yield game
        row = cursor.fetchone()
