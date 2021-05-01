import pytest

from catanatron_server import create_app
from catanatron_server.models import db


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # create the app with common test config
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})

    # create the database and load test data
    with app.app_context():
        db.init_app(app)
        db.create_all()

    yield app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_create_game_get_game_and_run_action(client):
    response = client.post("/api/games")
    response_json = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "game_id" in response_json
    game_id = response_json["game_id"]

    response = client.get(f"/api/games/{game_id}")
    response_json = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "tiles" in response_json

    response = client.post(f"/api/games/{game_id}/actions")
    response_json = response.get_json()
    assert response.status_code == 200
    assert response.is_json
    assert "tiles" in response_json


def test_game_not_exists(client):
    response = client.get("/api/games/123")
    assert response.status_code == 404
