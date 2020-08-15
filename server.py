import json

from flask import Flask, jsonify
from flask_cors import CORS

from models import generate_board

app = Flask(__name__)
CORS(app)


def serialize_tile(tile):
    return {"resource": None if tile.resource == None else tile.resource.value}


def serialize_port(port):
    return {"resource": None if port.resource == None else port.resource.value}


def serialize_board(board):
    return {
        "ports": [serialize_port(port) for port in board.ports],
        "tiles": [serialize_tile(tile) for tile in board.tiles],
        "numbers": board.numbers,
    }


@app.route("/board")
def board():
    board = generate_board()
    return jsonify(serialize_board(board))

