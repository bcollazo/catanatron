import pytest
import json
from catanatron.web import create_app
from catanatron.models.player import Color
from catanatron.game import (
    Game,
)  # Assuming Game can be imported for test setup/assertion
from catanatron.web.models import (
    db,
    GameState,
)  # For direct db interaction if needed for setup/teardown


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Setup an in-memory SQLite database for testing
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SECRET_KEY": "test",
        }
    )

    with app.app_context():
        db.create_all()

    yield app

    # Teardown: drop all tables after each test (optional, if tests are isolated)
    # with app.app_context():
    #     db.drop_all()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_post_game_endpoint(client):
    """Test creating a new game."""
    response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "game_id" in data
    # Further check: Ensure the game was actually created in the db
    with client.application.app_context():
        assert (
            db.session.query(GameState).filter_by(uuid=data["game_id"]).first()
            is not None
        )


def test_get_game_endpoint(client):
    """Test retrieving a specific game state."""
    # First, create a game to retrieve
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    game_id = json.loads(post_response.data)["game_id"]

    # Retrieve the initial state (state_index 0)
    response = client.get(f"/api/games/{game_id}/states/0")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "nodes" in data
    assert "edges" in data
    assert data["is_initial_build_phase"] is True
    assert data["winning_color"] is None


def test_get_latest_game_endpoint(client):
    """Test retrieving the latest game state."""
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    game_id = json.loads(post_response.data)["game_id"]

    response = client.get(f"/api/games/{game_id}/states/latest")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "nodes" in data
    assert "edges" in data
    assert data["is_initial_build_phase"] is True
    assert data["winning_color"] is None


def test_get_game_not_found(client):
    """Test retrieving a non-existent game."""
    response = client.get("/api/games/nonexistentgameid/states/0")
    assert response.status_code == 404


def test_post_action_bot_turn(client):
    """Test posting an action when it's a bot's turn."""
    # Create a game with at least one bot (RANDOM is a bot)
    post_response = client.post("/api/games", json={"players": ["RANDOM", "HUMAN"]})
    assert post_response.status_code == 200
    game_id = json.loads(post_response.data)["game_id"]

    data_before_res = client.get(f"/api/games/{game_id}/states/latest")
    data_before = json.loads(data_before_res.data)

    after_action_res = client.post(f"/api/games/{game_id}/actions", json={})
    assert after_action_res.status_code == 200
    data_after = json.loads(after_action_res.data)

    # Check if game state progressed, e.g., turn changed or actions list grew
    assert len(data_after["actions"]) > len(data_before["actions"])


def test_mcts_analysis_endpoint(client):
    """Test the MCTS analysis endpoint."""
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    game_id = json.loads(post_response.data)["game_id"]

    # Request MCTS analysis for the latest state
    response = client.get(f"/api/games/{game_id}/states/latest/mcts-analysis")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "probabilities" in data
    # Further checks on probabilities structure if known
    assert len(data["probabilities"]) == 2  # For two players


def test_mcts_analysis_game_not_found(client):
    """Test MCTS analysis for a non-existent game."""
    response = client.get("/api/games/nonexistent/states/nonexistent/mcts-analysis")
    assert response.status_code == 400


# def test_post_action_human_turn_valid_action(client, app):
#     """Test posting a valid action when it's a human player's turn."""
#     # Create a game where the first player is HUMAN
#     # Need to adjust player_factory or game setup if HUMAN isn't default first
#     # For now, let's assume we can make the first player HUMAN
#     # This might require a refactor in api.py or a more complex setup here.

#     # Simplified: Create a game, then manually update it to be a human's turn
#     # This is a bit of a hack for testing without deeper refactoring yet.
#     with app.app_context():
#         # Create a game with a human player
#         # The player_factory in api.py will create players based on the list.
#         # Let's make the first player HUMAN
#         game_players_config = ["HUMAN", "RANDOM"]
#         post_response = client.post("/api/games", json={"players": game_players_config})
#         assert post_response.status_code == 200
#         game_id = json.loads(post_response.data)["game_id"]

#         # Retrieve the game directly from DB to manipulate for testing
#         # This is usually not ideal but helps for targeted testing here.
#         from catanatron.web.models import get_game_state, upsert_game_state

#         game = get_game_state(game_id)
#         # Ensure it's human's turn (player 0, Color.RED by default)
#         # If game.state.current_player() is already human and not a bot, we are good.
#         # If not, this test setup needs to be more robust or the game logic needs to be invoked.

#         # For this example, let's assume the first player is HUMAN and it's their turn.
#         # We need a valid action. The simplest action is often 'END_TURN'.
#         # The structure of action_from_json and game.execute needs to be known.
#         # Let's assume an 'END_TURN' action looks like this:
#         valid_action_json = {
#             "type": "END_TURN",
#             "player_color": game.state.current_player().color.value,
#         }

#     action_response = client.post(
#         f"/api/games/{game_id}/actions", json=valid_action_json
#     )
#     assert action_response.status_code == 200
#     action_data = json.loads(action_response.data)
#     assert len(action_data["state"]["actions"]) > 0  # Check if an action was recorded


# Example of a test for player_factory if it were more complex or standalone
# from catanatron.web.api import player_factory # Assuming direct import is possible
# def test_player_factory():
#     player = player_factory(("CATANATRON", Color.RED))
#     assert isinstance(player, AlphaBetaPlayer)
#     assert player.color == Color.RED

#     player = player_factory(("RANDOM", Color.BLUE))
#     assert isinstance(player, RandomPlayer)
#     assert player.color == Color.BLUE

#     player = player_factory(("HUMAN", Color.ORANGE))
#     assert isinstance(player, ValueFunctionPlayer)
#     assert not player.is_bot
#     assert player.color == Color.ORANGE

#     with pytest.raises(ValueError):
#         player_factory(("INVALID_TYPE", Color.WHITE))


# Stress test endpoint is simple, just check if it runs
def test_stress_test_endpoint(client):
    response = client.get("/api/stress-test")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["winning_color"] is None
