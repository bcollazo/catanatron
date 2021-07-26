import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier

mpl.rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_feature_importances(X, y):
    columns = X.columns
    # Build a forest and compute the impurity-based feature importances
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(
            "%d. feature %s (%f)"
            % (f + 1, columns[indices[f]], importances[indices[f]])
        )

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(
        range(X.shape[1]),
        importances[indices],
        color="r",
        yerr=std[indices],
        align="center",
    )
    plt.xticks(range(X.shape[1]), [columns[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()


def train(X, y):
    """example of auto-sklearn for a classification dataset"""
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1
    )
    # define search
    model = AutoSklearnClassifier(
        time_left_for_this_task=30,
        # per_run_time_limit=30,
        # n_jobs=8,
    )
    # perform the search
    model.fit(X_train, y_train)
    # summarize
    print(model.sprint_statistics())
    # evaluate best model
    y_hat = model.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print("Accuracy: %.3f" % acc)

    model_path = Path("./catanatron/players/estimator.pickle").resolve()
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def plot_metrics(history):
    metrics = ["loss", "auc", "precision", "recall"]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[0],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "auc":
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
