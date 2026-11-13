"""K-nearest neighbors classifier for cardiovascular disease.

Predicts whether a patient has cardiovascular disease ("cardio") from eight
clinical measurements in the Kaggle Cardiovascular Disease dataset
(cardio_train.csv, 70,000 patients).

Written in Google Colab for the Eduexplora Stanford Artificial Intelligence
program (Brazil, January 2020), then cleaned up to run as a script.

Usage:
    python knn_heart.py [path/to/cardio_train.csv]
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["age", "weight", "ap_hi", "ap_lo",
            "cholesterol", "gluc", "smoke", "active"]
TARGET = "cardio"
SEED = 42


def load_data(path):
    """Read the semicolon-separated dataset.

    :param path: str, path to cardio_train.csv
    :returns: tuple of the feature table X and the target y
    """
    data = pd.read_csv(path, sep=";")
    return data[FEATURES].copy(), data[TARGET].copy()


def scale(X_train, X_test):
    """Min-max scale both sets using the training set's range only.

    Scaling puts every feature on 0-1, so no single measurement (such as
    age in days) dominates the distance between patients.

    :returns: tuple of the scaled training and test arrays
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def evaluate_k(k, X_train, y_train, X_test, y_test):
    """Train a KNN classifier with k neighbors and score it on the test set.

    :param k: int, number of neighbors
    :returns: tuple of accuracy and recall on the test set
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred), recall_score(y_test, y_pred)


def sweep_k(k_list, X_train, y_train, X_test, y_test):
    """Test accuracy for every k in k_list.

    :returns: list of accuracies, in the same order as k_list
    """
    return [evaluate_k(k, X_train, y_train, X_test, y_test)[0]
            for k in k_list]


def plot_accuracy(k_list, acc_list, out="accuracy_vs_k.png"):
    """Plot test accuracy against k and save it as an image.

    :returns: str, path of the saved image
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, acc_list, "ro--", linewidth=2)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.title("Test accuracy vs. number of neighbors")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def grid_search_k(X_train, y_train, k_values=range(3, 100, 2)):
    """Choose k by 5-fold cross-validated recall on the training set.

    Scaling sits inside the pipeline, so it is refit within each fold and
    no information leaks from the validation folds.

    :returns: tuple of the best k and its mean cross-validated recall
    """
    model = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
    grid = GridSearchCV(model,
                        param_grid={"kneighborsclassifier__n_neighbors":
                                    k_values},
                        scoring="recall", cv=5)
    grid.fit(X_train, y_train)
    return (grid.best_params_["kneighborsclassifier__n_neighbors"],
            grid.best_score_)


def main(path="cardio_train.csv"):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train_scaled, X_test_scaled = scale(X_train, X_test)

    acc, rec = evaluate_k(5, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"k = 5: accuracy {acc:.3f}, recall {rec:.3f}")

    k_list = list(range(3, 500, 2))
    acc_list = sweep_k(k_list, X_train_scaled, y_train,
                       X_test_scaled, y_test)
    best = max(acc_list)
    print(f"Best test accuracy {best:.3f} at k = "
          f"{k_list[acc_list.index(best)]}")
    print("Saved", plot_accuracy(k_list, acc_list))

    k, score = grid_search_k(X_train, y_train)
    print(f"Grid search (5-fold CV, recall): best k = {k}, "
          f"recall {score:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
