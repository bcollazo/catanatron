import os
import json
import pickle
from contextlib import contextmanager
from catanatron.json import GameEncoder

from catanatron.game import Game
from catanatron.state_functions import get_state_index
from sqlalchemy import MetaData, Column, Integer, String, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from flask_sqlalchemy import SQLAlchemy
from flask import abort

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
    @staticmethod
    def from_game(game: Game):
        state = json.dumps(game, cls=GameEncoder)
        # Use game.copy() to create a clean copy
        game_copy = game.copy()
        
        # Try standard pickle first (fastest and most compatible)
        try:
            pickle_data = pickle.dumps(game_copy, pickle.HIGHEST_PROTOCOL)
        except (TypeError, AttributeError) as e:
            # If standard pickle fails, try dill as fallback
            # This handles more complex objects but is slower
            try:
                import dill
                pickle_data = dill.dumps(game_copy)
            except ImportError:
                raise ValueError(
                    "Unable to pickle game. Standard pickle failed, and 'dill' is not installed. "
                    "Install dill for better serialization: pip install dill"
                ) from e
            except Exception as dill_error:
                # If even dill fails, provide helpful error message
                raise ValueError(
                    f"Unable to pickle game. Original error: {str(e)}. "
                    f"Dill also failed: {str(dill_error)}. "
                    "This may be due to unpickleable objects in player classes. "
                    "Ensure player classes implement __getstate__ and __setstate__ if needed."
                ) from e
        
        return GameState(
            uuid=game.id,
            state_index=get_state_index(game.state),
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
    # Try standard pickle first, fall back to dill if needed
    try:
        game = pickle.loads(result.pickle_data)  # type: ignore
    except (TypeError, pickle.UnpicklingError):
        # If standard unpickler fails, try dill
        try:
            import dill
            game = dill.loads(result.pickle_data)  # type: ignore
        except ImportError:
            raise ValueError(
                "Unable to unpickle game. "
                "Install 'dill' package for better lock handling: pip install dill"
            )
    return game
