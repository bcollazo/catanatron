---
icon: hexagon-check
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
