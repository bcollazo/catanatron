import os
import json
import pickle
from contextlib import contextmanager

from sqlalchemy import MetaData, Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from flask_sqlalchemy import SQLAlchemy

from catanatron.json import GameEncoder

# Using approach from: https://stackoverflow.com/questions/41004540/using-sqlalchemy-models-in-and-out-of-flask/41014157
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class GameState(Base):
    __tablename__ = "game_states"

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), nullable=False)
    state_index = Column(Integer, nullable=False)
    state = Column(String, nullable=False)
    pickle_data = Column(LargeBinary, nullable=False)

    # TODO: unique uuid and state_index


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
        session.close()


def create_game_state(game):
    state = json.dumps(game, cls=GameEncoder)
    pickle_data = pickle.dumps(game, pickle.HIGHEST_PROTOCOL)
    game.id, len(game.state.actions), state, pickle_data
    game_state = GameState(
        uuid=game.id,
        state_index=len(game.state.actions),
        state=state,
        pickle_data=pickle_data,
    )
    db.session.add(game_state)
    db.session.commit()
    return game_state


def get_game_state(game_id):
    result = (
        db.session.query(GameState)
        .filter_by(uuid=game_id)
        .order_by(GameState.state_index.desc())
        .first_or_404()
    )
    game = pickle.loads(result.pickle_data)
    return game
