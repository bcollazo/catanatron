import os
import json
import pickle

import psycopg2

from catanatron.json import GameEncoder

CREATE_TABLE_QUERY = """
    CREATE TABLE IF NOT EXISTS games (
        uuid            VARCHAR(36) NOT NULL UNIQUE, 
        state           JSON, 
        pickle_data     BYTEA
    )
    """
UPSERT_GAME_QUERY = """
    INSERT INTO games VALUES (%s, %s, %s)
        ON CONFLICT(uuid) 
        DO UPDATE SET 
            state=excluded.state,
            pickle_data=excluded.pickle_data;
    """
GET_GAME_QUERY = """SELECT pickle_data FROM games WHERE uuid = %s"""

connection = psycopg2.connect(
    user="catanatron",
    password="victorypoint",
    host="127.0.0.1",
    port="5432",
    database="catanatron_db",
)
cursor = connection.cursor()

cursor.execute(CREATE_TABLE_QUERY)


def save_game(uuid, game):
    state = json.dumps(game, cls=GameEncoder)
    pickle_data = pickle.dumps(game, pickle.HIGHEST_PROTOCOL)
    cursor.execute(
        UPSERT_GAME_QUERY,
        (str(uuid), state, pickle_data),
    )
    connection.commit()


def get_game(uuid):
    cursor.execute(GET_GAME_QUERY, (uuid,))
    row = cursor.fetchone()
    if row is None:
        return None

    pickle_data = row[0]
    game = pickle.loads(pickle_data)
    return game
