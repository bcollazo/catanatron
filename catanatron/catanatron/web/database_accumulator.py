from catanatron.game import GameAccumulator
from catanatron.web.models import database_session, upsert_game_state
from catanatron.web.utils import ensure_link


class StepDatabaseAccumulator(GameAccumulator):
    """
    Saves a game state to database for each tick.
    Slows game ~1s per tick.
    """

    def before(self, game):
        with database_session() as session:
            upsert_game_state(game, session)

    def step(self, game_before_action, action):
        with database_session() as session:
            upsert_game_state(game_before_action, session)

    def after(self, game):
        self.link = ensure_link(game, get_replay_link=True)


class DatabaseAccumulator(GameAccumulator):
    """Saves last game state to database"""

    def after(self, game):
        self.link = ensure_link(game)
