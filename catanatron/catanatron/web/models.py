import os
import json
import pickle
from contextlib import contextmanager
from catanatron.json import GameEncoder

from catanatron.game import Game
from catanatron.state_functions import get_state_index
from sqlalchemy import (
    MetaData,
    Column,
    Integer,
    String,
    LargeBinary,
    create_engine,
    and_,
    func,
    inspect,
    text,
)
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
    replay_source_folder = Column(String(1024), nullable=True)
    imported_at_utc = Column(String(64), nullable=True)

    # TODO: unique uuid and state_index
    @staticmethod
    def from_game(
        game: Game,
        replay_source_folder: str | None = None,
        imported_at_utc: str | None = None,
    ):
        state = json.dumps(game, cls=GameEncoder)
        pickle_data = pickle.dumps(game, pickle.HIGHEST_PROTOCOL)
        return GameState(
            uuid=game.id,
            state_index=get_state_index(game.state),
            state=state,
            pickle_data=pickle_data,
            replay_source_folder=replay_source_folder,
            imported_at_utc=imported_at_utc,
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


def upsert_game_state(
    game: Game,
    session_param=None,
    replay_source_folder: str | None = None,
    imported_at_utc: str | None = None,
):
    game_state = GameState.from_game(
        game,
        replay_source_folder=replay_source_folder,
        imported_at_utc=imported_at_utc,
    )
    session = session_param or db.session
    session.add(game_state)
    session.commit()
    return game_state


def ensure_game_state_metadata_columns():
    """Add replay metadata columns for existing DBs if missing."""
    inspector = inspect(db.engine)
    existing = {col["name"] for col in inspector.get_columns("game_states")}
    statements: list[str] = []
    if "replay_source_folder" not in existing:
        statements.append(
            "ALTER TABLE game_states ADD COLUMN replay_source_folder VARCHAR(1024)"
        )
    if "imported_at_utc" not in existing:
        statements.append("ALTER TABLE game_states ADD COLUMN imported_at_utc VARCHAR(64)")

    if not statements:
        return

    with db.engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


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
    game = pickle.loads(result.pickle_data)  # type: ignore
    return game


def _infer_us_color(payload: dict) -> str | None:
    bot_labels = payload.get("bot_labels") or {}
    for color, label in bot_labels.items():
        if isinstance(label, str) and "(US)" in label:
            return color

    colors = payload.get("colors") or []
    bot_colors = payload.get("bot_colors") or []
    # Fallback for Capstone 1v1 convention (our policy is BLUE bot).
    if "BLUE" in colors and "BLUE" in bot_colors:
        return "BLUE"
    return None


def _pip_value(number: int | None) -> int:
    pip_lookup = {
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        8: 5,
        9: 4,
        10: 3,
        11: 2,
        12: 1,
    }
    return pip_lookup.get(number, 0) if number is not None else 0


def _opening_pip_score(payload: dict, color: str | None, action_records: list) -> int | None:
    if color is None:
        return None
    settlement_nodes: list[int] = []
    for action_record in action_records:
        if not action_record or not action_record[0]:
            continue
        action = action_record[0]
        if action[0] == color and action[1] == "BUILD_SETTLEMENT":
            settlement_nodes.append(action[2])
            if len(settlement_nodes) >= 2:
                break
    if len(settlement_nodes) < 2:
        return None

    adjacent_tiles = payload.get("adjacent_tiles") or {}
    score = 0
    for node in settlement_nodes:
        for tile in adjacent_tiles.get(str(node), []):
            if tile.get("type") == "RESOURCE_TILE":
                score += _pip_value(tile.get("number"))
    return score


def _first_city_turn(action_records: list, us_color: str | None) -> int | None:
    if us_color is None:
        return None
    turns_elapsed = 0
    for action_record in action_records:
        if not action_record or not action_record[0]:
            continue
        action = action_record[0]
        if action[1] == "END_TURN":
            turns_elapsed += 1
        if action[0] == us_color and action[1] == "BUILD_CITY":
            return turns_elapsed
    return None


def list_replay_catalog(limit: int = 200):
    latest_by_uuid = (
        db.session.query(
            GameState.uuid.label("uuid"),
            func.max(GameState.state_index).label("max_state_index"),
        )
        .group_by(GameState.uuid)
        .subquery()
    )

    rows = (
        db.session.query(GameState)
        .join(
            latest_by_uuid,
            and_(
                GameState.uuid == latest_by_uuid.c.uuid,
                GameState.state_index == latest_by_uuid.c.max_state_index,
            ),
        )
        .order_by(GameState.id.desc())
        .limit(limit)
        .all()
    )

    catalog = []
    for row in rows:
        payload = json.loads(row.state)
        us_color = _infer_us_color(payload)
        colors = payload.get("colors") or []
        winning_color = payload.get("winning_color")
        # Catalog is intentionally for completed games only.
        if winning_color is None:
            continue
        action_records = payload.get("action_records") or []
        player_state = payload.get("player_state") or {}
        turn_count = sum(
            1
            for action_record in action_records
            if action_record and action_record[0] and action_record[0][1] == "END_TURN"
        )
        went_first = colors[0] == us_color if us_color and len(colors) > 0 else None
        won = winning_color == us_color if us_color and winning_color is not None else None
        us_index = colors.index(us_color) if us_color in colors else None
        opp_index = None
        opp_color = None
        if us_index is not None and len(colors) == 2:
            opp_index = 1 - us_index
            opp_color = colors[opp_index]

        us_final_vp = (
            player_state.get(f"P{us_index}_ACTUAL_VICTORY_POINTS")
            if us_index is not None
            else None
        )
        opp_final_vp = (
            player_state.get(f"P{opp_index}_ACTUAL_VICTORY_POINTS")
            if opp_index is not None
            else None
        )

        us_buy_dev = 0
        us_maritime_trades = 0
        us_build_city = 0
        us_build_settlement = 0
        us_play_knight = 0
        us_build_actions = 0
        us_trade_actions = 0
        us_dev_actions = 0
        us_robber_actions = 0
        us_total_actions = 0
        for action_record in action_records:
            if not action_record or not action_record[0]:
                continue
            action = action_record[0]
            action_color = action[0]
            action_type = action[1]
            if action_color != us_color:
                continue
            us_total_actions += 1
            if action_type == "BUY_DEVELOPMENT_CARD":
                us_buy_dev += 1
                us_dev_actions += 1
            elif action_type == "MARITIME_TRADE":
                us_maritime_trades += 1
                us_trade_actions += 1
            elif action_type == "BUILD_CITY":
                us_build_city += 1
                us_build_actions += 1
            elif action_type == "BUILD_SETTLEMENT":
                us_build_settlement += 1
                us_build_actions += 1
            elif action_type == "BUILD_ROAD":
                us_build_actions += 1
            elif action_type == "PLAY_KNIGHT_CARD":
                us_play_knight += 1
                us_dev_actions += 1
            elif action_type in ("PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING"):
                us_dev_actions += 1
            elif action_type in ("OFFER_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "CONFIRM_TRADE", "CANCEL_TRADE"):
                us_trade_actions += 1
            elif action_type == "MOVE_ROBBER":
                us_robber_actions += 1

        opening_pip_score = _opening_pip_score(payload, us_color, action_records)
        opp_opening_pip_score = _opening_pip_score(payload, opp_color, action_records)
        opening_pip_diff = (
            opening_pip_score - opp_opening_pip_score
            if opening_pip_score is not None and opp_opening_pip_score is not None
            else None
        )
        first_city_turn = _first_city_turn(action_records, us_color)

        catalog.append(
            {
                "game_id": row.uuid,
                "state_index": row.state_index,
                "turn_count": turn_count,
                "winner": winning_color,
                "us_color": us_color,
                "went_first": went_first,
                "won": won,
                "us_final_vp": us_final_vp,
                "opp_final_vp": opp_final_vp,
                "us_buy_dev": us_buy_dev,
                "us_maritime_trades": us_maritime_trades,
                "us_build_city": us_build_city,
                "us_build_settlement": us_build_settlement,
                "us_play_knight": us_play_knight,
                "us_opening_pip_score": opening_pip_score,
                "opp_opening_pip_score": opp_opening_pip_score,
                "opening_pip_diff": opening_pip_diff,
                "us_first_city_turn": first_city_turn,
                "us_action_build": us_build_actions,
                "us_action_trade": us_trade_actions,
                "us_action_dev": us_dev_actions,
                "us_action_robber": us_robber_actions,
                "us_action_total": us_total_actions,
                "replay_source_folder": row.replay_source_folder,
                "imported_at_utc": row.imported_at_utc,
            }
        )

    return catalog
