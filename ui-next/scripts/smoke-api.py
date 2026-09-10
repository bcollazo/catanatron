"""Exercise the existing Flask API without a running database or web server.

Run from the repository root with a Python environment containing .[web].
No production data is read or changed; the test uses an in-memory database.
"""
import json
import random
from collections import Counter
from pathlib import Path
from catanatron.web import create_app
from catanatron.web.models import get_game_state, upsert_game_state
from catanatron.models.enums import ActionPrompt, WOOD
from catanatron.models.actions import generate_playable_actions

random.seed(37)
app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
client = app.test_client()
counts = Counter()
states = []


def state_response(response):
    assert response.status_code == 200, response.data
    state = response.get_json()
    assert state["tiles"] and state["nodes"] and state["colors"]
    states.append(state)
    return state


for template in ["BASE", "MINI", "TOURNAMENT"]:
    response = client.post("/api/games", json={
        "players": ["HUMAN", "WEIGHTED_RANDOM", "RANDOM"],
        "map_template": template, "vps_to_win": 15,
        "discard_limit": 9, "friendly_robber": True,
    })
    assert response.status_code == 200, response.data
    game_id = response.get_json()["game_id"]
    state = state_response(client.get(f"/api/games/{game_id}/states/latest"))
    for _ in range(150):
        if state["winning_color"]:
            break
        before = state["state_index"]
        if state["current_color"] in state["bot_colors"]:
            response = client.post(f"/api/games/{game_id}/actions")
        else:
            options = state["current_playable_actions"]
            # Explore less common actions before ending the turn.
            action = min(options, key=lambda a: (a[1] == "END_TURN", counts[a[1]]))
            response = client.post(f"/api/games/{game_id}/actions", json=action)
        state = state_response(response)
        assert state["state_index"] == before + 1
        counts[state["action_records"][-1][0][1]] += 1
    first = state_response(client.get(f"/api/games/{game_id}/states/0"))
    assert first["state_index"] == 0 and first["action_records"] == []

# Exercise the discard prompt deterministically instead of relying on a lucky 7.
with app.app_context():
    game = get_game_state(game_id)
    index = next(i for i,p in enumerate(game.state.players) if not p.is_bot)
    game.state.current_player_index = index
    game.state.current_turn_index = index
    game.state.current_prompt = ActionPrompt.DISCARD
    game.state.is_discarding = True
    game.state.discard_counts = [2 if i == index else 0 for i in range(len(game.state.players))]
    game.state.player_state[f"P{index}_{WOOD}_IN_HAND"] = 2
    game.playable_actions = generate_playable_actions(game.state)
    # A new fixture ID avoids a duplicate index among already saved states.
    game.id = "discard-fixture"
    upsert_game_state(game)
state = state_response(client.get("/api/games/discard-fixture/states/latest"))
for remaining in [1, 0]:
    action = next(a for a in state["current_playable_actions"] if a[1] == "DISCARD_RESOURCE" and a[2] == WOOD)
    state = state_response(client.post("/api/games/discard-fixture/actions", json=action))
    assert state["current_discard_count"] == remaining
    counts["DISCARD_RESOURCE"] += 1
assert state["current_prompt"] == "MOVE_ROBBER"

# Catanatron is the API's alpha-beta bot. Verify its real action endpoint too.
game_id = client.post("/api/games", json={"players": ["CATANATRON", "RANDOM"]}).get_json()["game_id"]
state = state_response(client.post(f"/api/games/{game_id}/actions"))
assert state["state_index"] == 1
assert client.get("/api/games/not-a-game/states/latest").status_code == 404

# Save representative responses for the frontend validator smoke check.
output = Path(__file__).resolve().parents[1] / ".api-smoke.json"
output.write_text(json.dumps(states, separators=(",", ":")), encoding="utf-8")
print(f"API smoke passed: 3 maps, 3 bot types, {sum(counts.values())} moves, historical state retrieval, and 404 handling.")
print(dict(counts))
