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

    def step(self, game):
        with database_session() as session:
            upsert_game_state(game, session)


class DatabaseAccumulator(GameAccumulator):
    """Saves last game state to database"""

    def after(self, game):
        self.link = ensure_link(game)
