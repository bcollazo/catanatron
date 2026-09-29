"""Persisting games as a document plus what happened to it.

A game used to be stored by writing its whole document once per tick, so a
400-tick game cost megabytes and the history was re-serialized in every row.
Two things make that unnecessary:

- Every random outcome the engine produces is recorded in the action itself
  (the dice, the stolen resource, the drawn card), so re-applying the log
  rebuilds the game exactly -- see :mod:`tests.test_replay`.
- Replay never consults the players, so it does not matter what decided an
  action, or whether that thing could ever be made to decide it again.

So a game is one row holding the document it starts from and the document it
is at now, plus one small row per action. Earlier states are materialized by
replaying the log onto the first document; the current one is read straight
off the second, because that is what nearly every request asks for.

**No migration**: this reads neither the old ``game_states`` table nor
anything written before this change.
"""

import json
import os
from contextlib import contextmanager

from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.registry import REGISTRY
from catanatron.serialization import (
    SCHEMA_VERSION,
    action_record_from_json,
    action_record_to_json,
    state_from_json,
    state_to_json,
)
from catanatron.state_functions import get_state_index
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from flask_sqlalchemy import SQLAlchemy
from flask import abort

# Using approach from: https://stackoverflow.com/questions/41004540/using-sqlalchemy-models-in-and-out-of-flask/41014157
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class StoredGame(Base):
    """A game: where it started, who is playing, and where its rng is now.

    ``base_state`` is a full document, but only one per game -- whatever state
    the game was first saved at, which is normally its first tick. Everything
    after it lives in :class:`StoredAction`.
    """

    __tablename__ = "games"

    uuid = Column(String(64), primary_key=True)
    schema_version = Column(Integer, nullable=False)
    player_specs = Column(String, nullable=False)
    #: How many actions the base document already contains.
    base_index = Column(Integer, nullable=False)
    base_state = Column(String, nullable=False)
    #: How many actions the log holds. The game's current state_index.
    head_index = Column(Integer, nullable=False)
    #: The current state, overwritten on every write. Two reasons to keep it
    #: rather than replay for it: it carries the live random stream, which a
    #: log cannot rebuild (replay consumes no randomness, it re-applies
    #: recorded outcomes), and reading the latest state is what nearly every
    #: request wants, so it should not cost a replay of the whole game.
    head_state = Column(String, nullable=False)


class StoredAction(Base):
    """One entry of a game's log: ``[action, result]`` as JSON.

    The composite key is also the index every read needs, and it is what stops
    two concurrent writers from both appending the same position.
    """

    __tablename__ = "game_actions"

    uuid = Column(String(64), ForeignKey("games.uuid"), primary_key=True)
    index = Column(Integer, primary_key=True)
    payload = Column(String, nullable=False)


db = SQLAlchemy(metadata=metadata)


@contextmanager
def database_session():
    """Can use like:
    with database_session() as session:
        games = session.query(StoredGame).all()
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
    """Save a game: create its row the first time, then append what is new."""
    session = session_param or db.session
    head = get_state_index(game.state)
    document = json.dumps(state_to_json(game))
    stored = session.get(StoredGame, game.id)

    if stored is None:
        stored = StoredGame(
            uuid=game.id,
            schema_version=SCHEMA_VERSION,
            player_specs=json.dumps(
                [REGISTRY.spec_of(player) for player in game.state.players]
            ),
            base_index=head,
            base_state=document,
            head_index=head,
            head_state=document,
        )
        session.add(stored)

    for index in range(stored.head_index, head):
        record = game.state.action_records[index]
        session.add(
            StoredAction(
                uuid=game.id,
                index=index,
                payload=json.dumps(action_record_to_json(record)),
            )
        )

    stored.head_index = head
    stored.head_state = document
    session.commit()
    return stored


def get_game_state(game_id, state_index=None) -> Game | None:
    """Materialize a game at ``state_index``, or at its latest state.

    Only the latest state carries the live random stream, because only it can
    be played on. An earlier state is replayed exactly, but it inherits the
    stream the game had at its base, so treat it as read-only.
    """
    session = db.session
    stored = session.get(StoredGame, game_id)
    if stored is None:
        abort(404)
    if stored.schema_version != SCHEMA_VERSION:
        # There is no migration: a game written by an older document shape
        # cannot be rebuilt by this one.
        abort(410, description=f"game was stored by schema v{stored.schema_version}")

    target = stored.head_index if state_index is None else state_index
    if not stored.base_index <= target <= stored.head_index:
        abort(404)

    at_head = target == stored.head_index
    document = json.loads(stored.head_state if at_head else stored.base_state)

    specs = json.loads(stored.player_specs)
    colors = [Color[c] for c in document["colors"]]
    players = [REGISTRY.build(spec, color) for spec, color in zip(specs, colors)]
    game = state_from_json(document, players)
    if at_head:
        return game

    log = (
        session.query(StoredAction)
        .filter(
            StoredAction.uuid == game_id,
            StoredAction.index >= stored.base_index,
            StoredAction.index < target,
        )
        .order_by(StoredAction.index)
        .all()
    )
    session.commit()
    if len(log) != target - stored.base_index:
        # Replaying a hole would silently produce a state the game never had.
        abort(500, description=f"game {game_id} is missing actions before {target}")

    for entry in log:
        record = action_record_from_json(json.loads(entry.payload))
        game.execute(record.action, validate_action=False, action_record=record)
    return game
