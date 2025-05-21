---
icon: window
---

# Graphical User Interface

We provide a [docker-compose.yml](../../docker-compose.yml) with everything needed to play and watch games (useful for debugging). It contains all the web-server infrastructure needed to render a game in a browser.

<figure><img src="../.gitbook/assets/CatanatronUI (1).png" alt=""><figcaption><p>Catanatron Web UI</p></figcaption></figure>

To use, ensure you have [Docker Compose](https://docs.docker.com/compose/install/) installed, and run (from this repo's root):

```bash
docker compose up
```

You should now be able to visit [http://localhost:3000](http://localhost:3000/) and play!

You can also (in a new terminal window) install the `[web]` subpackage and use the `--db` flag to make the catanatron-play simulator save the game in the database for inspection via the web server.

```bash
pip install .[web]
catanatron-play --players=W,W,W,W --db --num=1
```

The link should be printed in the console.

{% hint style="info" %}
A great contribution would be to make the Web UI allow to step forwards and backwards in a game to inspect it (ala chess.com)!
{% endhint %}
