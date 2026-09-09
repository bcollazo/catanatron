"""Measure PR #386 snapshot construction and a persistent JSON echo process.

Fetch the two pinned PR files into target/pr386 first (see README.md).
Uses the PR's actual serializer/message builder against the unchanged base engine.
This isolates wire cost; it is not a benchmark of StdioPlayer or Rust JSON parsing.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import baseline
from baseline import HERE, bench, new_game


def load(name):
    spec = importlib.util.spec_from_file_location(
        "catanatron." + name, HERE / "target" / "pr386" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    load("serialization")
    protocol = load("protocol")
    game = new_game(0)
    snapshots = [game.copy()]
    while game.winning_color() is None and game.state.num_turns < 1000:
        game.play_tick()
        if len(game.state.action_records) == 512:
            snapshots.append(game.copy())
    snapshots.append(game.copy())
    worker = (
        "import sys,json\n"
        "for line in sys.stdin:\n"
        " m=json.loads(line)\n"
        " print(json.dumps({'action':m['playable_actions'][0]}),flush=True)\n"
    )
    result = []
    with subprocess.Popen([sys.executable, "-u", "-c", worker], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1) as child:
        for g in snapshots:
            make = lambda: json.dumps(protocol.decide_message(g, g.state.current_color())) + "\n"
            payload = make()
            assert g.playable_actions
            def echo():
                child.stdin.write(payload)
                child.stdin.flush()
                reply = json.loads(child.stdout.readline())
                assert reply["action"] == json.loads(payload)["playable_actions"][0]
            # Avoid counting the assertion's second parse in the timed version.
            echo()
            def exchange():
                child.stdin.write(payload)
                child.stdin.flush()
                return json.loads(child.stdout.readline())
            result.append(dict(history=len(g.state.action_records), bytes=len(payload.encode()),
                construction=bench("pr386_decide_construct_json", make, 50),
                persistent_echo=bench("prebuilt_json_echo_roundtrip", exchange, 50)))
        child.stdin.close()
        child.wait(timeout=5)
    (HERE / "protocol-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([dict(history=r["history"], bytes=r["bytes"],
        construction_us=r["construction"]["median"] / 1000,
        echo_us=r["persistent_echo"]["median"] / 1000) for r in result], indent=2))


if __name__ == "__main__":
    main()
