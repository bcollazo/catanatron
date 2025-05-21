---
icon: square-terminal
---

# Command Line Interface

### Basic Usage Examples

*   Run 1,000 games with Random Players

    ```bash
    catanatron-play --num 1000 --players R,R,R,R 
    ```
*   Run 10 1v1 games between VictoryPointPlayer and ValueFunctionPlayer

    ```bash
    catanatron-play --num 10 --players VP,F 
    ```
*   Pit 3-player Catan with basically no discard limit

    ```bash
    catanatron-play --num 1 --players W,F,AB:2 --config-discard-limit 999
    ```
*   Play 1v1 against ValueFunctionPlayer until 15 points with discard limit set to 9 (would not recommend :sweat\_smile:; its much better to play using the GUI)

    {% code overflow="wrap" %}
    ```bash
    catanatron-play --num 1 --players F,H --quiet --config-discard-limit 9 --config-vps-to-win 15 
    ```
    {% endcode %}

### Using Different Players

Many of the Bots are ready to be used from the CLI using the `--players` flag. Specify players by using `<id>:<param1>:<param2>:...` syntax. For example,

* `G:10` is GreedyPlayoutsPlayer with rollouts = 10
* `AB:3:True` is AlphaBetaPlayer with 3 ply look-ahead and prunning turned on
* ...

For more information, use:

```bash
catanatron-play --help-players
```

<figure><img src="../.gitbook/assets/Screenshot 2025-05-21 111738.png" alt=""><figcaption><p>catanatron-play --help-players output</p></figcaption></figure>

### Saving Game Results

You can save the output of games using the `--output` and `--output-format` flags. For example:&#x20;

#### JSON Format

```bash
catanatron-play --num 5 --players F,F,R,R --output data/ --output-format json
```

Inspecting the `data` directory you should see all 5 games.

```
data/
├── 2aba0ec4-51e7-47b6-b5a9-113874b2addf.json
├── 353a541b-59d2-4eb8-85ea-d6eb0d162e7f.json
├── 3c5f37ce-a73c-4180-91fd-28bf9f6dacd2.json
├── e31b00a3-845c-4bfd-92bd-c4a42cbafa27.json
└── ee547837-d5e2-4d9b-8420-583b2fdd793d.json

1 directory, 5 files
```

#### CSV

With `--output-format csv` you can generate the CSVs suitable for Machine Learning and / or Data Analysis.

```bash
catanatron-play --num 5 --players F,F,R,R --output data/ --output-format csv
```

This generates 4 GZIP CSVs:

* **samples.csv.gz:** One row per game state at each [ply](../core-concepts.md).
* **actions.csv.gz:** Integers representing actions taken by each player at each ply.
* **rewards.csv.gz:** Some common returns and rewards with which to label how effective actions taken by bots where. See Reinforcement Learning section on [openapi.md](../advanced/openapi.md "mention")
* **main.csv.gz:** Simply a concatenation of the above 3 CSVs

Using `--output-format csv` will continuously simply append to these 4 files more and more plys even if from different games.

#### Parquet

[Parquet format](https://parquet.apache.org/) is also supported. One file per game is generated (like in JSON format)

```bash
catanatron-play --num 5 --players F,R --output data/ --output-format parquet
```

#### Include Board Tensors

Representing the Catan Board State for Machine Learning purposes is challenging using tabular data. The best we can often do is have many features like `TILE11_IS_BRICK` or `TILE3_PROBA` to represent what resource and what number lie in what tile. Similarly columns like `NODE9_P2_CITY` and so on to represent if Player 2 has a City in Node 9.

To aid in this regard, you can also generate a 3D Tensor for each game state capturing the spatial relationship of these features. You can generate it when using `--output-format parquet` or `--output-format csv` by using the `--include-board-tensors` flag.

For example:

```bash
catanatron-play --num 5 --players AB:2,AB:2 --output data/ --output-format csv --include-board-tensor
```

Its a dedicated flag since it makes simulations a bit slower.
