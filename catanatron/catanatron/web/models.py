import os
import json
from contextlib import contextmanager

from catanatron.game import Game
from catanatron.registry import REGISTRY
from catanatron.serialization import SCHEMA_VERSION, state_from_json, state_to_json
from catanatron.state_functions import get_state_index
from sqlalchemy import MetaData, Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from flask_sqlalchemy import SQLAlchemy
from flask import abort

from catanatron.models.player import Color

# Using approach from: https://stackoverflow.com/questions/41004540/using-sqlalchemy-models-in-and-out-of-flask/41014157
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class GameState(Base):
    """One persisted game state.

    The game is stored as JSON that completely defines it (see
    :mod:`catanatron.serialization`); the players are stored separately as
    specs, so that a saved game never depends on pickled bot code.
    """

    __tablename__ = "game_states"

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), nullable=False)
    state_index = Column(Integer, nullable=False)
    schema_version = Column(Integer, nullable=False)
    state = Column(String, nullable=False)
    player_specs = Column(String, nullable=False)

    # TODO: unique uuid and state_index
    @staticmethod
    def from_game(game: Game):
        specs = [REGISTRY.spec_of(player) for player in game.state.players]
        return GameState(
            uuid=game.id,
            state_index=get_state_index(game.state),
            schema_version=SCHEMA_VERSION,
            state=json.dumps(state_to_json(game)),
            player_specs=json.dumps(specs),
        )

    def to_game(self) -> Game:
        doc = json.loads(self.state)
        specs = json.loads(self.player_specs)
        colors = [Color[c] for c in doc["colors"]]
        players = [REGISTRY.build(spec, color) for spec, color in zip(specs, colors)]
        return state_from_json(doc, players)


db = SQLAlchemy(metadata=metadata)


@contextmanager
def database_session():
    """Can use like:
    with database_session() as session:
        game_states = session.query(GameState).all()
    """
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://catanatron:victorypoint@127.0.0.1:5432/catanatron_db",
    )
    engine = create_engine(database_url)
    session = Session(engine)
    try:
        yield session
    finally:
        session.expunge_all()
        session.close()


def upsert_game_state(game: Game, session_param=None):
    game_state = GameState.from_game(game)
    session = session_param or db.session
    session.add(game_state)
    session.commit()
    return game_state


def get_game_state(game_id, state_index=None) -> Game | None:
    """
    Returns the game from database.
    """
    if state_index is None:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id)
            .order_by(GameState.state_index.desc())
            .first()
        )
        if result is None:
            abort(404)
    else:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id, state_index=state_index)
            .first()
        )
        if result is None:
            abort(404)
    db.session.commit()
    return result.to_game()
