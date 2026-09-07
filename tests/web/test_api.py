import pytest
import json
from catanatron.web import create_app
from catanatron.web.models import db, GameState, get_game_state


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
    response = client.post("/api/games", json={"players": ["R", "R"]})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "game_id" in data
    # Further check: Ensure the game was actually created in the db
    with client.application.app_context():
        assert (
            db.session.query(GameState).filter_by(uuid=data["game_id"]).first()
            is not None
        )


def test_post_game_endpoint_accepts_custom_config(client):
    response = client.post(
        "/api/games",
        json={
            "players": ["W", "CATANATRON"],
            "map_template": "MINI",
            "vps_to_win": 15,
            "discard_limit": 12,
            "friendly_robber": True,
        },
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    with client.application.app_context():
        game = get_game_state(data["game_id"])
        assert game.friendly_robber is True

    state_response = client.get(f"/api/games/{data['game_id']}/states/latest")
    assert state_response.status_code == 200
    state_data = json.loads(state_response.data)
    land_tiles = [
        tile for tile in state_data["tiles"] if tile["tile"]["type"] != "WATER"
    ]
    assert len(land_tiles) == 7


def test_post_game_endpoint_rejects_invalid_config(client):
    response = client.post(
        "/api/games",
        json={
            "players": ["R"],
            "map_template": "INVALID",
            "vps_to_win": 25,
            "discard_limit": 2,
        },
    )
    assert response.status_code == 400


def test_get_game_endpoint(client):
    """Test retrieving a specific game state."""
    # First, create a game to retrieve
    post_response = client.post("/api/games", json={"players": ["R", "R"]})
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
    post_response = client.post("/api/games", json={"players": ["R", "R"]})
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
    post_response = client.post("/api/games", json={"players": ["R", "HUMAN"]})
    assert post_response.status_code == 200
    game_id = json.loads(post_response.data)["game_id"]

    data_before_res = client.get(f"/api/games/{game_id}/states/latest")
    data_before = json.loads(data_before_res.data)

    after_action_res = client.post(f"/api/games/{game_id}/actions", json={})
    assert after_action_res.status_code == 200
    data_after = json.loads(after_action_res.data)

    # Check if game state progressed, e.g., turn changed or actions list grew
    assert len(data_after["action_records"]) > len(data_before["action_records"])


def test_repeated_bot_actions_advance_latest_state(client):
    """Latest state should keep advancing across persisted bot turns."""
    post_response = client.post("/api/games", json={"players": ["R", "R"]})
    assert post_response.status_code == 200
    game_id = json.loads(post_response.data)["game_id"]

    latest_before = json.loads(client.get(f"/api/games/{game_id}/states/latest").data)
    first_tick = json.loads(client.post(f"/api/games/{game_id}/actions", json={}).data)
    second_tick = json.loads(client.post(f"/api/games/{game_id}/actions", json={}).data)
    latest_after = json.loads(client.get(f"/api/games/{game_id}/states/latest").data)

    assert first_tick["state_index"] == latest_before["state_index"] + 1
    assert second_tick["state_index"] == first_tick["state_index"] + 1
    assert len(second_tick["action_records"]) == len(first_tick["action_records"]) + 1
    assert latest_after["state_index"] == second_tick["state_index"]
    assert latest_after["action_records"] == second_tick["action_records"]


def test_mcts_analysis_endpoint(client):
    """Test the MCTS analysis endpoint."""
    post_response = client.post("/api/games", json={"players": ["R", "R"]})
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


# Stress test endpoint is simple, just check if it runs
def test_stress_test_endpoint(client):
    response = client.get("/api/stress-test")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["winning_color"] is None


# ===== registry-backed players =====
def test_get_players_endpoint_lists_seatable_players(client):
    response = client.get("/api/players")
    assert response.status_code == 200
    payload = json.loads(response.data)

    by_key = {entry["key"]: entry for entry in payload}
    assert {"CATANATRON", "HUMAN", "R", "W", "AB"} <= set(by_key)
    assert by_key["HUMAN"]["is_bot"] is False
    assert by_key["CATANATRON"]["is_bot"] is True
    # friendly labels, not internal class names
    assert by_key["HUMAN"]["name"] == "Human"
    assert by_key["R"]["name"] == "Random"


def test_get_players_endpoint_has_no_duplicate_aliases(client):
    """Two keys for the same class would show twice in the UI dropdown."""
    payload = json.loads(client.get("/api/players").data)
    names = [entry["name"] for entry in payload]
    assert len(names) == len(set(names))


def test_get_players_endpoint_publishes_param_schema(client):
    payload = json.loads(client.get("/api/players").data)
    alphabeta = next(entry for entry in payload if entry["key"] == "AB")
    depth = next(param for param in alphabeta["params"] if param["name"] == "depth")
    assert depth["type"] == "int"
    assert depth["default"] == 2


def test_post_game_accepts_params_in_a_player_spec(client):
    response = client.post(
        "/api/games", json={"players": ["R", "AB:depth=1:prunning=true"]}
    )
    assert response.status_code == 200
    game_id = json.loads(response.data)["game_id"]

    with client.application.app_context():
        game = get_game_state(game_id)
        bot = next(
            p for p in game.state.players if type(p).__name__ == "AlphaBetaPlayer"
        )
        assert bot.params.depth == 1
        assert bot.params.prunning is True


def test_post_game_accepts_structured_player_specs(client):
    response = client.post(
        "/api/games",
        json={"players": ["R", {"key": "AB", "params": {"depth": 1}}]},
    )
    assert response.status_code == 200
    game_id = json.loads(response.data)["game_id"]
    with client.application.app_context():
        game = get_game_state(game_id)
        bot = next(
            p for p in game.state.players if type(p).__name__ == "AlphaBetaPlayer"
        )
        assert bot.params.depth == 1


def test_post_game_rejects_unknown_player_with_400(client):
    response = client.post("/api/games", json={"players": ["R", "NOSUCHBOT"]})
    assert response.status_code == 400


def test_post_game_rejects_unknown_param_with_400(client):
    response = client.post("/api/games", json={"players": ["R", "AB:dpeth=3"]})
    assert response.status_code == 400

