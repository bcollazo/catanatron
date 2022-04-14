## Here a log of results as they are found:

- We were able to create a Python package that implements the Game's core logic.
  No fancy networkX or numpy models, just good old Python objects and readable
  algorithms. Fully tested!

- We build a UI to go along with it. Not only useful for debugging, but if robot
  comes to life, we may want to allow users to play against it!

- WeightedRandomPlayer consistently wins to RandomPlayer.

- ValueFunction exposed that copying game with `deepcopy` is slow. Potential
  road block for Tree-Search algorithms (e.g. mini-max). Played OK. Too slow
  to benchmark really. Computed features.

- Started playing with DRL approaches. Threw away useful features (like production)
  and used RAW representation (someone said automatic feature engineering).

- Started with Policy learning using "Cross Entropy Method" here: ...
  Noticed we face class-imbalance. Infra was still inmature.
  Wasn't getting a meaningful LOSS, so scrapped it.

- Started with Deep Q-Learning. Seemed we were able to achieve a reasonable LOSS
  (MAE=0.05) on a "Discounted Return" with `DISCOUNT_FACTOR=0.999`. Played
  terrible tho. Not better than random (maybe actually worst).

- Tried Deep Q-Learning only on top-examples. Nothing interesting.

- Tried Deep Learning ValueFunction, and doing greedy algorithm on it. Won 63%
  of the games against RandomPlayers. Single-epoch, 32-batch, 75,000 samples.
  RAW Features. Games take around 30s tho.

  ```
  VRLPlayer:Foo[RED](Version1) [35] ████████████████████████████████████████
  RandomPlayer:Bar[BLUE] [22] █████████████████████████▏
  RandomPlayer:Baz[ORANGE] [10] ███████████▍
  WeightedRandomPlayer:Qux[WHITE] [33] █████████████████████████████████████▊
  ```

- Using hand-crafted production and expansion features, we achieved 0.15 MAE.

- Using AutoKeras on hand-crafted production and expansion features yielded
  something like 0.01 MSE. But in performance, still played just like Random
  tho. Did not perform better.

- Got very good performance using VICTORY_POINTS_RETURN with 30GB data set,
  batch size = 256, batchnormalization => dense(64) => dense(1) value func.
  MSE loss, adam with lr=0.001,clipnorm=1. 1 epoch. Achieved ~2.0 MAE.

- First BoardTensor model seems to overfit (places on same nodes). ~200,000 params
  with less than 1,000,000 samples. Still, after normalizing wins 44/100 games
  against random players. NEXT: Try simpler model and more data.

- Second BoardTensor model (adam(0.001,clipnorm=1),
  input1 = 21x11x16 => batchnorm => 1 filter-cnn kernel=3 linear => flatten
  input2 = num-features
  model = input1 + input2 => 64 relu => 32 relu => 1 linear.
  MSE loss on VP RETURN. Batch=256, epoch=1, 2M samples.
  Performed comparable to simple 2-layer on 1M samples.
  - RandomPlayer:Foo[RED] [14] ███████████████▏
  - RandomPlayer:Bar[BLUE] [15] ████████████████▎
  - VRLPlayer:Baz[ORANGE](models/vp-big-256-64) [37] ████████████████████████████████████████
  - TensorRLPlayer:Qux[WHITE](tensor-model-normalized) [34] ████████████████████████████████████▊

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 21, 11, 16)] 0
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 21, 11, 16)   64          input_1[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 19, 9, 1)     145         batch_normalization[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 171)          0           conv2d[0][0]
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 74)]         0
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 245)          0           flatten[0][0]
                                                                 input_2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 64)           15744       concatenate[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 32)           2080        dense[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            33          dense_1[0][0]
==================================================================================================
Total params: 18,066
Trainable params: 18,034
Non-trainable params: 32
__________________________________________________________________________________________________
```

- We prototyped the Playouts Theorem. It indeed gives a good value function to
  play out random games from a given state. Pretty stable at at least 50 games.
  25 games there was some flakyness... (could stable out given we do this)
  many times. Takes around 0.3 secs to play a random game to completion.

- MCTS (doing 25 playouts for each possible action) wins almost every
  game against randoms, but it takes a while. And it seems some high-branching
  factor situations make it take longer (e.g. monopoly play, late road building).
  Here some results:
  AVG Turns: 301.6
  AVG Duration: 1024.1514820575715
  RandomPlayer:Foo[RED] [0]
  RandomPlayer:Bar[BLUE] [0]
  RandomPlayer:Baz[ORANGE] [0]
  MCTSPlayer:Qux[WHITE] [5] ████████████████████████████████████████

- It seems learning the simple problem of whether a player produces OWS
  works well with 1 filter of size 3, linear, unit_norm, and l2-reg. =>
  flat => 1 dense sigmoid. Adding a 0.2 dropout before 1 dense sigmoid
  seemed to help a lot to not overfit. Next steps would like to separate
  edges and nodes from channels (it seems it confuses these two), and add
  max-pooling in an attempt to learn hierarchy (OWS-node, variaty-node, etc...).
  Does EPOCHS help? EPOCHS=2 overfits. Well we get TRAIN_ACC=90%+,TEST_ACC=60%.
- No class weight, simple 1 3x3 filter CNN seems to learn, just needs more data.
- No class weight, using autokerastuner arrives at 96 neurons 1-layer 0.0001 LR.
  Achieves 57% accuracy (using log-loss as objective).

- Practicing on simple problem of given a 2x2 array, trying to learn to count
  the number of 1s. it performs well. With num-samples=100,000, epoch=1,
  2-32-relu layers gets mae: 0.6571, val_mae: 0.0301. Default Adam, with loss=mse.
  With 1 more order of magnitude of data, we can get 1 order of magnitude less error. (val_mae: 0.0034)
  Next steps:

  - Use auto keras to see how much better can we get.
    This achieved val_mae: 0.0496. Not much significant improvement.
  - Use CNN to see how it affects counting.
    This improves model to 0.009 mae (with 100,000 samples). note params increased
    from 4k to 65k (from normal NN to 32-filter CNN). 1-filter got val_mae: 1.3479.
    PROBLEM SOLVED!
  - Next steps: complicated by adding planes of random data. Does it confuse NN?
    Adding 3 extra random data layers, did not confuse NN. Still achieves mae~0.004.
  - Next step: Can we learn AND'ed points across channels?
    Seems that CNN behaves better here (1 3x3 filter) as expected since problem
    is translation invariant. AutoKeras yields a very deep model. DNN get to ~80%
    val_accuracy. CNN gets to 90%+. They seem sensitive to initial values tho.
  - Next step: Does adding random layers disturb the AND learning?
    Yes. It seems CNN have trouble with noisy channels.
  - Next step: What if we add some 2s as well... (here we prove we could learn count-vp-problem)
    Yes. Can learn (without noise channels), to val_mae: 0.1147.

- We fixed a bug in the tensor board representation. Using only the first 13 planes
  allows a CNN to learn if a player has OWS successfully (88% val_accuracy). Results:

```
Trial 27 Complete [00h 10m 37s]
val_accuracy: 0.8878348469734192

Best val_accuracy So Far: 0.888144850730896
Total elapsed time: 05h 03m 43s

Search: Running Trial #28

Hyperparameter |Value |Best Value So Far
filters |9 |17
num_flat_layers |3 |1
learning_rate |0.0001 |0.001
units_0 |16 |24
units_1 |24 |32
units_2 |8 |None
tuner/epochs |10 |4
tuner/initial_e...|0 |2
tuner/bracket |0 |2
tuner/round |0 |1
```

- Regroup on Jan 2021.

  - We achieve mae: 5.4591e-11, val_mae: 2.4891e-04 with a deep model on RepA,
    and DISCOUNTED_RETURN. But model doesnt perform well in games. 50 game
    epochs, aiming to use 1K games, but early stopping.
    loss: 1.2714e-20 - mae: 5.4591e-11 - val_loss: 3.2321e-07 - val_mae: 2.4807e-04
  - Similar results with Rep B. But it played terribly...

- Online MCTS is looking promising. Did a first run with a 1-neuron model (normalizing
  features), playouts=25, training everytime there are >1K samples in batches of 64.
  There was a bug in the data as well (was creating samples like root-node => mcts result
  for all branches, instead of branch-node => mcts result). It didn't win games,
  but there seemed to be VP improvement from 2 VP avg to 5 VP avg, in 100 games.

- A heuristic player plays better than WeightedRandom (but we have a memory leak).
  Memory leak was due to @functools.lru with copied boards.

- Simple MiniMax Algorithm takes too long... ~20segs per decision,
  which makes for 30 min games. Even if de-bugged the implementation seems pretty slow.
  As such, alpha-pruning doesnt seem wont help much (even if cuts time by half).

- Using PyPy (removing TF and other Python 3.9 features), speeds up playing random
  games by around 33%s. (1000s games w/CPython3.9 took 2m52s and 1000s games
  w/PyPy3.7 took 1m52s).

- Looked at RL Python Frameworks. keras-rl doesn't work with TF2.0, same with
  stable-baselines. Many undocumented or hard to use. Best one seems to be
  tensorforce. TF-Agents seems possible, but pretty raw API right now.
  Hard to use tensorforce because not clear how to specify Rep B mixed model.

- VictoryPoint player is better than Random (much better) and WeightedRandom (not so much).

- Where able to do a somewhat strong NN with clipnorm=1, 2 layers, 32 and 8 neurons
  batch norm of inputs and batches of 32. 7,000,000 samples divided in 10 epochs.
  Still this player doesnt beat hand-heuristic. (ValueFunctionPlayer). This used
  DISCOUNTED_RETURN. MSE as loss and LR=0.0001.
  `loss: 1.5084e-07 - mae: 2.1945e-04 - val_loss: 4.2677e-05 - val_mae: 5.0571e-04`

- Did a RandomForest, but data seems wrong. Played very poorly. Tried again with
  all Rep 1 features; no dice. Hard to analyze... Probably overfitted. Keeps ending turn.

- Scikit Regression player doesnt win, because just gets a lot of wheat production,
  builds roads to extend as much as they can (dont hold to build settlements),
  place robber on themselves, dont trade (same value); no hand-diversity features.

- Greedy (even when budgeted to run same num playouts as MCTS) does better than
  MCTS.

- ValueFunction 2.0 (using EFFECTIVE_PRODUCTION and REACHABILITY features) plays
  pretty strongly. Better than Greedy=25 and MCTS=100.

- AlphaBeta(depth=2) plays stronger than ValueFunction 2.0. Using VF2.0 as heuristic.

- Tried Bayesian Optimization to search better weights for hand-crafted heuristic,
  but seems too slow and inefective.

- Note: In a random game, 38.6% of actions are roll (action=0), 27% are end turn (action=5557).

## Future Work:

- Should probably try to do NN player with better tuning (more layers, different label).

- Interested in seeing MCTS play in real life. Should compare G10 to MCTS50

- Work on performance.

- Generate data as 1 Big CSV per representation. (samples + labels), (bt + labels)

- Idea: Make test-evaluation framework. Use gaussian optimization to find best weights.

- Attempt DQN Tutorial with TF2.0 Summary Module, to see progress. If works,
  adapt to Catan-DQN Tutorial.

- Paperspace: Dockerfile: Install catanatron, and Pipenv dependencies, so that we can
  experiment/play.py, have a Game in the Overview.ipyn and run a catan-dqn.py job faster
  (should only do when can confirm Data and DQN approach improves...).

- Simplify ROBBER_ACTIONS to "Play Knight" to player with most points.

- Learn from offline MCTS data, using Rep B. (Seems slow to generate MCTS labeled data).

- Use tensorforce (since TF2.0 compatible and DQN Agent will be bug-free).
  Seems hard because API is not too easy.

- Learn an offline DQN (using VPs) with Rep B. That is, use Reward as label (
  so that early moves don't get a 1)

- Consider implementing AlphaBetaPruning to speed up game. Although
  it seems speedup at best might be 1/2 the time, which still doesn't cut it.
  Unless we speed up game copying...

- Actually use MCTS result to improve network. Do online so that we can
  see progress.

  - Add epsilon-playing to ensure we are exploring space. (Well, I think the fact
    that we have random players fixed does this job for now, since we collect
    samples from their perspective as well).

- Play a game against Greedy player. Play against Greedy with look-ahead.

- Idea: Sum up "enemy features". Make it look like its 1 enemy.

- Try Online - DQN approach (using PythonProgramming Tutorial).
- Use Value-Estimator with a tree-lookahead alpha-beta pruning.
- Try CNN-action space. (i.e. BUILD_SETTLEMENT at 3 means plane-board-tensor)
- Try policy-learning and q-learning on simpler action space.
- Try Cross-Entropy approach (using only top X features and dropping END TURNs).
- Try AlphaZero single-neural network learning (state) => (value, action).

  - Use in tree-search (take N top actions to explore via RollOuts).

- Try putting in features back. Is this VPlayer better than Raw Features one?
- Can autokeras use tf.Dataset?
- Using autokeras with whole 1Gb dataset is better?
- Does playing against your V1 self, training on that, improve?
- Try Q-Learning but, iterate on playing, learning, playing, learning... e-greedy.

### Performance Improvements

- Separate immutable state from Board (Map?), so that copying is a lot faster, and
  can cache functions (say node-production) globally (across game copies).

### Toy problems

- Idea: hot-encode 5 layers per player (one per resource) to denote income and buildings.
  then use 3D convolution of size WxHx5 and stride=5 (to not overlap)
- An easier problem would be to count house/city VPs (only uses 1 plane). count-vp-problem.
- Next medium problem will be, guess wheat production (to see if combining the two planes).

## Appendix

To install auto-keras and auto-sklearn:

```
auto-sklearn = "*"
emcee = "*"
pyrfr = "*"
scikit-optimize = "*"
pydoe = "*"
autokeras = "*"
kerastuner = {git = "https://github.com/keras-team/keras-tuner.git", ref = "1.0.2rc4"}
```

## Learnings

- Basic tf.Tensor manipulations
- Visualize data with `tfdv.visualize_statistics(stats)`
- Basic toy problems on Keras.
- Used auto-keras, auto-scikit.
- Basic how CNNs work. How LSTM (RNNs in general) work.
- Epochs, steps, generator datasets. GZIP compression.\
- Paperspace.
- DQN Algorithm.
- Noise can mislead a NN.

## Catan State Space Analysis

- Each tile can be any resource = 19. So 19! resource-tile decisions.
- Each tile must be one of the numbers. So 18! (desert has no number)
- There are 9 ports so: 9! ways of ordering them.
  Finally, there are 19! \* 18! \* 9! boards in Catan. So like 10^38 boards.

Configurations states inside it are upper bounded by:

- Each player has at most 5 houses. 54 choose 5 are ways of putting all 5 houses.
  54 choose 4 are ways of putting 4 houses. Sum\_{i in [0, 5]} (54 choose i) = 3.5M.
  Actually, better ignore colors and consider all 20 houses. So:
  `sum([comb(54, i) for i in range(21)]) = 683689022885551`.
  Actually, including cities, then there are 9 pieces per player that can be in board.
  `sum([comb(54, i) for i in range(4*9 + 1)]) = 17932673125137243`
- There are 14 roads per player so 56 roads in total. 72 possible locations so:
  `sum([comb(72, i) for i in range(56)]) = 4722360822358286581300`

If we include number of cards that makes the state space much much bigger,
but in practice its a lot less (its rare for a player to have 20+ cards). So
just using the Board States we see there are:
10^38 boards. Each with 17932673125137243 \* 4722360822358286581300 configurations,
which is almost like 10^37, so we are talking around 10^68 just possible board-states.

In terms of cards-in-hand state. Assuming on average players have 5 cards in hand,
then out of the 19*5 resource cards we start with, we are talking about:
`comb(19*4, 20) = 1090013712241956540` (or 10^18).

Grand-ballpark estimate is ~**10^100** states in Catan. Chess is 10^123.
Go is 10^360. Number of atoms in the Universe is 10^78.

### Branching Factor

The following stats suggests the decision tree players navigate usually has
around ~5 branches, and very few times something huge like 279 branches
(trading options after a monopoly(?), late road-building card(?)).

```
Branching Factor Stats:
count    94010.000000
mean         4.643772
std          7.684072
min          1.000000
25%          1.000000
50%          1.000000
75%          5.000000
max        279.000000
```

### Average Num Turns and Num Decisions

1000 Random Games result in:

```
AVG Ticks: 959.049
AVG Turns: 275.261
AVG Duration: 0.18782643032073976
AVG VPS: RandomPlayer:Foo[RED] 5.851
AVG VPS: RandomPlayer:Bar[BLUE] 5.871
AVG VPS: RandomPlayer:Baz[ORANGE] 5.766
AVG VPS: RandomPlayer:Qux[WHITE] 6.086
```

So each player makes around 70 decisions per game. Averages around 5.9 VPs per game.

### Performance Bits

- Creating an array with `np.array` is much faster than `tf.convert_to_tensor` and
  `pd.DataFrame` or `pd.Series`.

```
In [1]: timeit.timeit("np.array(array)", setup="import numpy as np; array = list([i for i in range(1000)])", number=1000)
Out[1]: 0.055867002985905856
In [2]: timeit.timeit("tf.convert_to_tensor(array)", setup="import tensorflow as tf; array = list([i for i in range(1000)])", number=1000)
Out[2]: 0.20465714897727594
In [3]: timeit.timeit("pd.DataFrame(array)", setup="import pandas as pd; array = list([i for i in range(1000)])", number=1000)
Out[3]: 0.333771405974403
```

- Game.copy() took 45% of the time when using ValueFunctionPlayer (5.7s).
- 3/21/21 playable_actions is 50% of time when running random 1v1s.
  Accounts for 7s in a 14s 100-game run.

Time Estimates:

- AB2 takes 0.01 seconds per tick.
