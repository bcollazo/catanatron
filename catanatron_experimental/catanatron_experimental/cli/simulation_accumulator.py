import os
import sys

# Add all necessary paths to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_core'))

try:
    from catanatron.game import GameAccumulator
except ImportError:
    # Fallback definition if import fails
    class GameAccumulator:
        def before(self, game):
            pass
            
        def step(self, game, action):
            pass
            
        def after(self, game):
            pass


class SimulationAccumulator(GameAccumulator):
    def before_all(self):
        """Called before all games in a catanatron-play simulation."""
        pass

    def after_all(self):
        """Called after all games in a catanatron-play simulation."""
        pass
