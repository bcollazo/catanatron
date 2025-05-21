---
icon: code
---

# Architecture Overview

The code is divided in three main components (folders):

* **catanatron**: The pure python implementation of the game logic. Uses `networkx` for fast graph operations. Is pip-installable (see [pyproject.toml](../../pyproject.toml)) and can be used as a Python package. See the documentation for the package here: [https://catanatron.readthedocs.io/](https://catanatron.readthedocs.io/).
  * **catanatron.web**: An extension package (optionally installed) that contains a Flask web server in order to serve game states from a database to a Web UI. The idea of using a database, is to ease watching games played in a different process. It defaults to using an ephemeral in-memory sqlite database. Also pip-installable with `pip install catanatron[web]`.
  * **catanatron.gym**: Gymnasium interface to Catan. Includes a 1v1 environment against a Random Bot and a vector-friendly representations of states and actions. This can be pip-installed independently with `pip install catanatron[gym]`, for more information see [catanatron/gym/README.md](../../catanatron/catanatron/gym/).
* **catantron\_experimental**: A collection of unorganized scripts with contain many failed attempts at finding the best possible bot. Its ok to break these scripts. Its pip-installable. Exposes a `catanatron-play` command-line script that can be used to play games in bulk, create machine learning datasets of games, and more!
* **ui**: A React web UI to render games. This is helpful for debugging the core implementation. We decided to use the browser as a randering engine (as opposed to the terminal or a desktop GUI) because of HTML/CSS's ubiquitousness and the ability to use modern animation libraries in the future ([https://www.framer.com/motion/](https://www.framer.com/motion/) or [https://www.react-spring.io/](https://www.react-spring.io/)).
