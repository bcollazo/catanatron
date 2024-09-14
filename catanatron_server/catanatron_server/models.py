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

    @staticmethod
    def from_game(game):
        print("Serializing game state...")
        state = json.dumps(game, cls=GameEncoder)
        try:
            pickle_data = pickle.dumps(game, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error pickling object: {e}")
            for name, value in vars(game).items():
                try:
                    pickle.dumps(value, pickle.HIGHEST_PROTOCOL)
                except Exception as sub_e:
                    print(f"Cannot pickle attribute {name}: {sub_e}")
            raise e
        return GameState(
            uuid=game.id,
            state_index=len(game.state.actions),
            state=state,
            pickle_data=pickle_data,
        )


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


def upsert_game_state(game, session_param=None):
    game_state = GameState.from_game(game)
    session = session_param or db.session
    session.add(game_state)
    session.commit()
    return game_state


def get_game_state(game_id, state_index=None):
    if state_index is None:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id)
            .order_by(GameState.state_index.desc())
            .first_or_404()
        )
    else:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id, state_index=state_index)
            .first_or_404()
        )
    db.session.commit()
    game = pickle.loads(result.pickle_data)
    return game
