---
icon: code
---

# Contributing

Any and all contributions are more than welcome!

## Running Tests

To develop for Catanatron, install the development dependencies and use the following test suite:

```bash
pip install ".[web,gym,dev]"
coverage run --source=catanatron -m pytest tests/ && coverage report
```

Or you can run the suite in watch-mode with:

```bash
ptw --ignore=tests/integration_tests/ --nobeep
```

## Architecture

The code is divided in three main components (folders):

* **catanatron**: The pure python implementation of the game logic. Uses `networkx` for fast graph operations. It is pip-installable (see [pyproject.toml](../pyproject.toml)) and can be used as a Python package. The implementation of this follows the idea of Game Trees (see [https://en.wikipedia.org/wiki/Game\_tree](https://en.wikipedia.org/wiki/Game_tree)) so that it lends itself for Tree-Searching Bots and Reinforcement Learning Environment Loops. Every "ply" is advanced with the `.play_tick` function. See more on Code Documentation site: [https://catanatron.readthedocs.io/](https://catanatron.readthedocs.io/)
  * **catanatron.web**: An extension package (optionally installed) that contains a Flask web server in order to serve game states from a database to a Web UI. The idea of using a database, is to ease watching games played in a different process. It defaults to using an ephemeral in-memory sqlite database. Also pip-installable with `pip install catanatron[web]`.
  * **catanatron.gym**: Gymnasium interface to Catan. Includes a configurable 1v1 environment and a vector-friendly representations of states and actions. This can be pip-installed independently with `pip install catanatron[gym]`, for more information see [catanatron/gym/README.md](../catanatron/catanatron/gym/).
  * **catanatron.cli**: A rich-powered CLI that enables the `catanatron-play` console script. Can be used to play games in bulk, create machine learning datasets of games, and more!
* **catantron\_experimental**: A collection of unorganized scripts with contain many failed attempts at finding the best possible bot. Its ok to break these scripts. Its pip-installable. 
* **ui**: A React web UI to render games. This is helpful for debugging the core implementation. We decided to use the browser as a randering engine (as opposed to the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness and the ability to use modern animation libraries in the future ([https://www.framer.com/motion/](https://www.framer.com/motion/) or [https://www.react-spring.io/](https://www.react-spring.io/)).

## Running Components Individually

As an alternative to running the project with Docker, you can run the web client and server in two separate tabs.

### React App

```bash
cd ui/
npm install
npm start
```

This can also be run via Docker independently (after building):

```bash
docker build -t bcollazo/catanatron-react-ui:latest ui/
docker run -it -p 3000:3000 bcollazo/catanatron-react-ui
```

### Flask Web Server

Ensure you are inside a virtual environment with all dependencies installed and
&#x20;use `flask run`. This will use SQLite by default.

```bash
pip install -e .[web]
FLASK_DEBUG=1 FLASK_APP=catanatron.web/catanatron.web flask run
```

This can also be run via Docker independently (after building):

```bash
docker build -t bcollazo/catanatron-server:latest . -f Dockerfile.web
docker run -it -p 5001:5001 bcollazo/catanatron-server
```

## Useful Commands

These are other potentially useful commands while developing catanatron

#### TensorBoard

For watching training progress, use `keras.callbacks.TensorBoard` and open TensorBoard:

```bash
tensorboard --logdir logs
```

#### Docker GPU TensorFlow

```bash
docker run -it tensorflow/tensorflow:latest-gpu-jupyter bash
docker run -it --rm -v $(realpath ./notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

#### Testing Performance

```bash
pyinstrument -r html --from-path catanatron-play --players AB:2,AB:2
```

```bash
python -m cProfile -o profile.pstats examples/play_batch_example.py
snakeviz profile.pstats
```

```bash
pytest --benchmark-compare=0001 --benchmark-compare-fail=mean:10% --benchmark-columns=min,max,mean,stddev
```

#### Head Large Datasets with Pandas

```python
import pandas as pd
x = pd.read_csv("data/mcts-playouts-labeling-2/labels.csv.gzip", compression="gzip", iterator=True)
x.get_chunk(10)
```

Building Sphinx Code Documentation Site

```bash
pip install -r docs/requirements.txt
sphinx-quickstart docs
sphinx-apidoc -o docs/source catanatron
sphinx-build -b html docs/source/ docs/build/html
```

#### Publishing to PyPi (Outdated)

catanatron Package

```bash
make build PACKAGE=catanatron
make upload PACKAGE=catanatron
make upload-production PACKAGE=catanatron
```

catanatron\_gym Package

```bash
make build PACKAGE=catanatron_gym
make upload PACKAGE=catanatron_gym
make upload-production PACKAGE=catanatron_gym
```

## Ideas for Contribution

* Improve `catanatron` package running time performance.
  * Continue refactoring the State to be more and more like a primitive `dict` or `array`. (Copies are much faster if State is just a native python object).
  * Move RESOURCE to be ints. Python `enums` turned out to be slow for hashing and using.
  * Move the `.action_records` action log concept to the Game class. This way MCTS algorithms that just need copy games for the purposes of rollouts, don't need to pay for copying the action_records, but AlphaBeta players can still use the log for undoing actions.
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
  * A Terminal UI? (for ease of debugging)
