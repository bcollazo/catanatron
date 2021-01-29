# This implements the Cross-Entropy Method for learning a Policy.
# It plays N games, selects the top 20% of games, tries to learn S => A
# mapping from there. Repeats.
N = 5  # 10 gives like ~1,600 PolicyLearning samples... or 10,000 QLearning samples.
K = 1
BASE_RETURN = 1000  # minimum G_t to consider for Policy Learning.


def find_player_by_class(game, player_class):
    return next(player for player in game.players if player.__class__ == player_class)


def build_policy_network():
    # Policy Network S => A, needs 221,394 params
    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]
    # not normalizing b.c. init random weights should output random anyways
    inputs = keras.Input(shape=(NUM_FEATURES,))
    outputs = keras.layers.Dense(32, activation=tf.nn.relu)(inputs)
    outputs = keras.layers.Dense(units=ACTION_SPACE_SIZE, activation=tf.nn.softmax)(
        outputs
    )
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        metrics=METRICS,
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )
    return model


# model = (
#     build_policy_network()
#     if RECREATE
#     else keras.models.load_model(NETWORK_MODEL_PATH)
# )
# if RECREATE:
#     model.save(NETWORK_MODEL_PATH)
#     print("Created model at:", NETWORK_MODEL_PATH)

# ===== POLICY LEARNING
# Compute class_weights
# ocurrances = Y.sum(axis=0)
# weights = (ocurrances.sum() / ocurrances).replace([np.inf, -np.inf], 0)
# class_weight = weights.to_dict()
# num_chunks = len(X) // CHUNK_SIZE
# for i, (X, Y) in enumerate(
#     zip(np.array_split(X, num_chunks), np.array_split(Y, num_chunks))
# ):
#     print("Training on batch", i)
#     scalar_metrics = model.train_on_batch(X, Y, class_weight=class_weight)
#     print(scalar_metrics)
# model.save(MODEL_PATH)
# print("Updated model at:", MODEL_PATH)
