---
icon: download
---

# Installation

### Command Line Interface Installation

1.  Clone the repository:

    ```bash
    git clone git@github.com:bcollazo/catanatron.git
    cd catanatron/
    ```
2.  Create a virtual environment (requires Python 3.11 or higher)&#x20;

    ```bash
    python -m venv venv
    source ./venv/bin/activate
    ```
3.  Install dependencies

    ```bash
    pip install -e .
    ```
4.  (Optional) Install developer and advanced dependencies&#x20;

    ```bash
    pip install -e .[web,gym,dev]
    ```

### Graphical User Interface Installation

1. Ensure you have Docker installed ([https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/))
2.  Run the `docker-compose.yaml` in the root folder of the repo:

    ```bash
    docker compose up
    ```
3. Visit [http://localhost:3000](http://localhost:3000/) in your browser!
