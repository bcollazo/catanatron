---
icon: code
---

# Contributing

Any and all contributions are more than welcome!

### Running Tests

To develop for Catanatron, install the development dependencies and use the following test suite:

```bash
pip install .[web,gym,dev]
coverage run --source=catanatron -m pytest tests/ && coverage report
```

Or you can run the suite in watch-mode with:

```bash
ptw --ignore=tests/integration_tests/ --nobeep
```

### Architecture

The code is divided in three main components (folders):

* **catanatron**: The pure python implementation of the game logic. Uses `networkx` for fast graph operations. It is pip-installable (see [pyproject.toml](../pyproject.toml)) and can be used as a Python package. See the documentation for the package here: [https://catanatron.readthedocs.io/](https://catanatron.readthedocs.io/). The implementation of this follows the idea of Game Trees (see [https://en.wikipedia.org/wiki/Game\_tree](https://en.wikipedia.org/wiki/Game_tree)) so that it lends itself for Tree-Searching Bots and Reinforcement Learning Environment Loops. Every "ply" is advanced with the `.play_tick` function. See more on Code Documentation site: [https://catanatron.readthedocs.io/](https://catanatron.readthedocs.io/)
  * **catanatron.web**: An extension package (optionally installed) that contains a Flask web server in order to serve game states from a database to a Web UI. The idea of using a database, is to ease watching games played in a different process. It defaults to using an ephemeral in-memory sqlite database. Also pip-installable with `pip install catanatron[web]`.
  * **catanatron.gym**: Gymnasium interface to Catan. Includes a 1v1 environment against a Random Bot and a vector-friendly representations of states and actions. This can be pip-installed independently with `pip install catanatron[gym]`, for more information see [catanatron/gym/README.md](../catanatron/catanatron/gym/).
* **catantron\_experimental**: A collection of unorganized scripts with contain many failed attempts at finding the best possible bot. Its ok to break these scripts. Its pip-installable. Exposes a `catanatron-play` command-line script that can be used to play games in bulk, create machine learning datasets of games, and more!
* **ui**: A React web UI to render games. This is helpful for debugging the core implementation. We decided to use the browser as a randering engine (as opposed to the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness and the ability to use modern animation libraries in the future ([https://www.framer.com/motion/](https://www.framer.com/motion/) or [https://www.react-spring.io/](https://www.react-spring.io/)).

### Ideas for Contributions

* Improve `catanatron` package running time performance.
  * Continue refactoring the State to be more and more like a primitive `dict` or `array`. (Copies are much faster if State is just a native python object).
  * Move RESOURCE to be ints. Python `enums` turned out to be slow for hashing and using.
  * Move the `.actions` action log concept to the Game class. (to avoid copying when copying State)
  * Remove `.current_prompt`. It seems its redundant with (is\_moving\_knight, etc...) and not needed.
* Improve AlphaBetaPlayer
  * Explore and improve prunning
  * Use Bayesian Methods or [SPSA](https://www.chessprogramming.org/SPSA) to tune weights and find better ones.
* Research!
  * Deep Q-Learning
  * Simple Alpha Go
  * Try Tensorforce with simple action space.
  * Try simple flat CSV approach but with AlphaBeta-generated games.
* Features
  * Continue implementing actions from the UI (not all implemented).
  * Chess.com-like UI for watching game replays (with Play/Pause and Back/Forward).
  * A terminal UI? (for ease of debugging)
