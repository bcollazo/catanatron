import pytest
import json

from catanatron_server import create_app
from catanatron_server.models import db


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # create the app with common test config
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})

    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_create_game_get_game_and_run_action(client):
    response = client.post(
        "/api/games",
        data=json.dumps({"players": ["RANDOM", "HUMAN"]}),
        content_type="application/json",
    )
    response_json = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "game_id" in response_json
    game_id = response_json["game_id"]

    response = client.get(f"/api/games/{game_id}/states/latest")
    response_json = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "tiles" in response_json

    response = client.get(f"/api/games/{game_id}/states/0")
    state_zero = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "tiles" in state_zero
    assert response_json == state_zero

    response = client.post(f"/api/games/{game_id}/actions")
    response_json = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "tiles" in response_json

    response = client.get(f"/api/games/{game_id}/states/latest")
    response_json = response.get_json()
    response = client.get(f"/api/games/{game_id}/states/0")
    state_zero = response.get_json()
    assert response_json != state_zero


def test_game_not_exists(client):
    response = client.get("/api/games/123")
    assert response.status_code == 404
