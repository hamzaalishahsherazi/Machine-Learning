"""
Project: Handwritten Digit Recognition with K-Nearest Neighbors (from scratch)

Idea: "Tell me who your neighbours are and I'll tell you who you are."
Every digit image is 8x8 = 64 pixel values, i.e. a point in 64-D space.
To recognise a new digit, we find the K training images closest to it
and let them vote on the answer.

What this project covers:
  1. KNN classifier written from scratch (vectorised with NumPy)
  2. Three distance metrics: Euclidean, Manhattan, Minkowski (p=3)
  3. Uniform voting vs distance-weighted voting
  4. Choosing the best K with 5-fold cross-validation
  5. Final test accuracy, confusion matrix, per-digit accuracy
  6. Comparison with scikit-learn's KNeighborsClassifier
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
#                  KNN from scratch
# ============================================================

class MyKNN:
    def __init__(self, k=3, metric="euclidean", weighted=False, p=3):
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.p = p

    def fit(self, X, y):
        # KNN is a "lazy learner": training = just storing the data
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self

    def _distances(self, X):
        """Distance from every test point to every training point -> (n_test, n_train)."""
        diff = X[:, None, :] - self.X_train[None, :, :]
        if self.metric == "euclidean":       # straight-line distance
            return np.sqrt(np.sum(diff ** 2, axis=2))
        if self.metric == "manhattan":       # "city block" distance
            return np.sum(np.abs(diff), axis=2)
        if self.metric == "minkowski":       # general form, p=1 Manhattan, p=2 Euclidean
            return np.sum(np.abs(diff) ** self.p, axis=2) ** (1 / self.p)
        raise ValueError("unknown metric")

    def predict(self, X, batch=200):
        preds = []
        for start in range(0, len(X), batch):          # batches keep memory low
            d = self._distances(X[start:start + batch])
            nearest = np.argsort(d, axis=1)[:, : self.k]
            for row, idx in enumerate(nearest):
                labels = self.y_train[idx]
                if self.weighted:                      # closer neighbours get bigger votes
                    w = 1 / (d[row, idx] + 1e-9)
                else:
                    w = np.ones(self.k)
                votes = {c: w[labels == c].sum() for c in np.unique(labels)}
                preds.append(max(votes, key=votes.get))
        return np.array(preds)


# ============================================================
#                  Helpers
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def cross_val_accuracy(model, X, y, folds=5, seed=0):
    """Split data into 'folds' parts, train on 4, test on 1, repeat, average."""
    idx = np.random.default_rng(seed).permutation(len(X))
    parts = np.array_split(idx, folds)
    scores = []
    for i in range(folds):
        test = parts[i]
        train = np.concatenate([parts[j] for j in range(folds) if j != i])
        scaler = StandardScaler().fit(X[train])
        model.fit(scaler.transform(X[train]), y[train])
        scores.append(accuracy(y[test], model.predict(scaler.transform(X[test]))))
    return np.mean(scores)


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def show_digit(image):
    """Print an 8x8 digit as text art."""
    chars = " .:-=+*#%@"
    for row in image.reshape(8, 8):
        print("      " + "".join(chars[int(v / 16 * 9)] * 2 for v in row))


# ============================================================
#                  MAIN
# ============================================================

if __name__ == "__main__":
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    print("=" * 60)
    print("KNN - Handwritten Digit Recognition")
    print("=" * 60)
    print(f"Images: {len(X)} (train {len(X_train)}, test {len(X_test)}), "
          f"pixels per image: {X.shape[1]}, classes: 0-9")

    # ---------- Step 1: choose K, metric and voting with cross-validation ----------
    print("\nStep 1: 5-fold cross-validation on the training set")
    print(f"{'K':>3} | {'Euclidean':>9} | {'Manhattan':>9} | {'Minkowski p=3':>13} | {'Euclid+weighted':>15}")
    results = {}
    for k in [1, 3, 5, 7, 9, 11]:
        row = {}
        for label, kwargs in [("euclidean", {"metric": "euclidean"}),
                              ("manhattan", {"metric": "manhattan"}),
                              ("minkowski", {"metric": "minkowski"}),
                              ("weighted", {"metric": "euclidean", "weighted": True})]:
            row[label] = cross_val_accuracy(MyKNN(k=k, **kwargs), X_train, y_train)
            results[(k, label)] = row[label]
        print(f"{k:>3} | {row['euclidean']:>9.4f} | {row['manhattan']:>9.4f} | "
              f"{row['minkowski']:>13.4f} | {row['weighted']:>15.4f}")

    (best_k, best_setting), best_cv = max(results.items(), key=lambda t: t[1])
    print(f"\nBest setting: K={best_k}, {best_setting} (CV accuracy {best_cv:.4f})")

    # ---------- Step 2: train final model, test on unseen data ----------
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    kwargs = {"metric": "euclidean", "weighted": True} if best_setting == "weighted" \
        else {"metric": best_setting}
    model = MyKNN(k=best_k, **kwargs).fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    sk_metric = "euclidean" if best_setting == "weighted" else best_setting
    sk = KNeighborsClassifier(n_neighbors=best_k, metric=sk_metric, p=3,
                              weights="distance" if best_setting == "weighted" else "uniform")
    sk_acc = sk.fit(Xtr, y_train).score(Xte, y_test)

    print("\nStep 2: Final test results")
    print(f"  My KNN accuracy      : {accuracy(y_test, y_pred):.4f}")
    print(f"  Sklearn KNN accuracy : {sk_acc:.4f}")

    # ---------- Step 3: where does it go wrong? ----------
    cm = confusion_matrix(y_test, y_pred, 10)
    print("\nStep 3: Confusion matrix (rows = true digit, columns = predicted)")
    print("       " + " ".join(f"{d:>3}" for d in range(10)))
    for d in range(10):
        print(f"  {d:>3}  " + " ".join(f"{v:>3}" for v in cm[d]))

    print("\nAccuracy per digit:")
    for d in range(10):
        print(f"  digit {d}: {cm[d, d] / cm[d].sum():.2%}")

    # ---------- Step 4: show a few predictions ----------
    print("\nStep 4: Sample predictions")
    wrong = np.where(y_pred != y_test)[0]
    samples = [0, 1] + list(wrong[:1])
    for i in samples:
        status = "CORRECT" if y_pred[i] == y_test[i] else "WRONG"
        print(f"\n  True: {y_test[i]}   Predicted: {y_pred[i]}   [{status}]")
        show_digit(X_test[i])
