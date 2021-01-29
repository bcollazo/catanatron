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

## Future Work:

### Toy problems

- Understand if mixed-data works. To practice mixed-data learning. Practice:
  - action => cost mapping.
  - state => victory points plain.
- Idea: hot-encode 5 layers per player (one per resource) to denote income and buildings.
  then use 3D convolution of size WxHx5 and stride=5 (to not overlap)
- Understand if 2D filters mix layers.
- An easier problem would be to count house/city VPs (only uses 1 plane). count-vp-problem.
- Next medium problem will be, guess wheat production (to see if combining the two planes).
- Does adding extra features distract the network?
- Use KerasTuner.
- Next steps would like to separate
  edges and nodes from channels (it seems it confuses these two), and add
  max-pooling in an attempt to learn hierarchy (OWS-node, variaty-node, etc...).

### Actual Catan

- Try CNN-action space. (i.e. BUILD_SETTLEMENT at 3 means plane-board-tensor)
- Try better board_tensor representation. (different channels, more 0s)
- Try policy-learning and q-learning on simpler action space.
- Try Cross-Entropy approach (using only top X features and dropping END TURNs).
- Try AlphaZero single-neural network learning (state) => (value, action).

  - Use in tree-search (take N top actions to explore via RollOuts).

- Understand how to improve tree-search approaches.

- Try putting in features back. Is this VPlayer better than Raw Features one?
- Can autokeras use tf.Dataset?
- Using autokeras with whole 1Gb dataset is better?
- Does playing against your V1 self, training on that, improve?
- Try Q-Learning but, iterate on playing, learning, playing, learning... e-greedy.

Performance:

- Consider memoizing feature vector computation (many stay the same). Would need
  serializable / primitive datastructure state.
- Faster game.copy()

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
- Epochs, steps, generator datasets. GZIP compression.
