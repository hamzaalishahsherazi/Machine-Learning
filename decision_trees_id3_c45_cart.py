"""
Project: Comparing Decision Tree Algorithms - ID3 vs C4.5 vs CART
(all three written from scratch in Python)

All three trees play "20 questions" with the data, but they differ in
HOW they choose the question and HOW MANY answers each question has:

  ID3   - picks the question with the highest INFORMATION GAIN (entropy drop).
          Opens one branch for every answer. Works only with categories.
  C4.5  - ID3's upgrade. Uses GAIN RATIO, which stops it favouring features
          with many values. Also handles numbers by finding a cut point.
  CART  - uses the GINI INDEX and only ever asks yes/no questions
          (two branches). Handles numbers and categories.

Part 1: Play Tennis dataset (categorical) - all three algorithms, step by step
Part 2: Iris dataset (numeric) - all three algorithms, accuracy compared
"""

import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ============================================================
#                  Impurity measures
# ============================================================

def entropy(labels):
    p = np.array(list(Counter(labels).values())) / len(labels)
    return -np.sum(p * np.log2(p))


def gini(labels):
    p = np.array(list(Counter(labels).values())) / len(labels)
    return 1 - np.sum(p ** 2)


def split_info(groups, n):
    """How 'spread out' a split is. Many small branches -> big value."""
    p = np.array([len(g) for g in groups if len(g)]) / n
    return -np.sum(p * np.log2(p))


def majority(labels):
    return Counter(labels).most_common(1)[0][0]


# ============================================================
#                  Base tree (shared code)
# ============================================================
# Node formats:
#   leaf       : {"leaf": label}
#   multi-way  : {"feature": f, "branches": {value: node}, "majority": m}
#   binary     : {"feature": f, "op": "<=" or "==", "value": v, "left": node, "right": node}

class BaseTree:
    def __init__(self, max_depth=None, numeric=()):
        self.max_depth = max_depth if max_depth is not None else 99
        self.numeric = set(numeric)          # names of numeric features

    def fit(self, rows, labels, features):
        self.root = self._build(rows, labels, list(features), 0)
        return self

    def _build(self, rows, labels, features, depth):
        if len(set(labels)) == 1 or not features or depth >= self.max_depth:
            return {"leaf": majority(labels)}
        split = self._best_split(rows, labels, features)
        if split is None:
            return {"leaf": majority(labels)}
        node = self._make_node(split, rows, labels, features, depth)
        # simple pruning: if every branch gives the same answer, the question is useless
        kids = list(node["branches"].values()) if "branches" in node else [node["left"], node["right"]]
        if all("leaf" in k for k in kids) and len({k["leaf"] for k in kids}) == 1:
            return kids[0]
        return node

    def _binary_node(self, f, op, v, rows, labels, features, depth):
        go_left = [self._test(r[f], op, v) for r in rows]
        L = [(r, l) for r, l, g in zip(rows, labels, go_left) if g]
        R = [(r, l) for r, l, g in zip(rows, labels, go_left) if not g]
        return {"feature": f, "op": op, "value": v,
                "left": self._build([r for r, _ in L], [l for _, l in L], features, depth + 1),
                "right": self._build([r for r, _ in R], [l for _, l in R], features, depth + 1)}

    def _multiway_node(self, f, rows, labels, features, depth):
        rest = [x for x in features if x != f]   # a category is used once per path
        node = {"feature": f, "branches": {}, "majority": majority(labels)}
        for v in sorted(set(r[f] for r in rows)):
            sub = [(r, l) for r, l in zip(rows, labels) if r[f] == v]
            node["branches"][v] = self._build([r for r, _ in sub], [l for _, l in sub],
                                              rest, depth + 1)
        return node

    @staticmethod
    def _test(x, op, v):
        return x <= v if op == "<=" else x == v

    @staticmethod
    def _thresholds(values):
        u = sorted(set(values))
        return [(a + b) / 2 for a, b in zip(u[:-1], u[1:])]

    def predict_one(self, row, node=None):
        node = node or self.root
        while "leaf" not in node:
            if "branches" in node:
                child = node["branches"].get(row[node["feature"]])
                if child is None:                    # unseen category
                    return node["majority"]
                node = child
            else:
                node = node["left"] if self._test(row[node["feature"]], node["op"],
                                                  node["value"]) else node["right"]
        return node["leaf"]

    def predict(self, rows):
        return [self.predict_one(r) for r in rows]

    def count_leaves(self, node=None):
        node = node or self.root
        if "leaf" in node:
            return 1
        kids = node["branches"].values() if "branches" in node else [node["left"], node["right"]]
        return sum(self.count_leaves(k) for k in kids)

    def show(self, node=None, indent="    "):
        node = node or self.root
        if "leaf" in node:
            print(f"{indent}-> {node['leaf']}")
        elif "branches" in node:
            for v, child in node["branches"].items():
                if "leaf" in child:
                    print(f"{indent}{node['feature']} = {v} -> {child['leaf']}")
                else:
                    print(f"{indent}{node['feature']} = {v}:")
                    self.show(child, indent + "    ")
        else:
            f, op, v = node["feature"], node["op"], node["value"]
            v_txt = f"{v:.2f}" if isinstance(v, float) else v
            neg = ">" if op == "<=" else "!="
            for test, child in ((f"{f} {op} {v_txt}", node["left"]),
                                (f"{f} {neg} {v_txt}", node["right"])):
                if "leaf" in child:
                    print(f"{indent}{test} -> {child['leaf']}")
                else:
                    print(f"{indent}{test}:")
                    self.show(child, indent + "    ")


# ============================================================
#                  ID3  -  Information Gain, multi-way
# ============================================================

class ID3(BaseTree):
    def scores(self, rows, labels, features):
        out = {}
        for f in features:
            groups = [[l for r, l in zip(rows, labels) if r[f] == v] for v in set(r[f] for r in rows)]
            out[f] = entropy(labels) - sum(len(g) / len(labels) * entropy(g) for g in groups)
        return out

    def _best_split(self, rows, labels, features):
        s = self.scores(rows, labels, features)
        best = max(s, key=s.get)
        return best if s[best] > 0 else None

    def _make_node(self, f, rows, labels, features, depth):
        return self._multiway_node(f, rows, labels, features, depth)


# ============================================================
#                  C4.5  -  Gain Ratio, multi-way + numeric
# ============================================================

class C45(BaseTree):
    def _feature_score(self, rows, labels, f):
        """Returns (gain_ratio, gain, threshold or None) for one feature."""
        n, base = len(labels), entropy(labels)
        if f in self.numeric:
            best = (0, 0, None)
            for t in self._thresholds([r[f] for r in rows]):
                L = [l for r, l in zip(rows, labels) if r[f] <= t]
                R = [l for r, l in zip(rows, labels) if r[f] > t]
                gain = base - (len(L) * entropy(L) + len(R) * entropy(R)) / n
                si = split_info([L, R], n)
                if si > 0 and gain / si > best[0]:
                    best = (gain / si, gain, t)
            return best
        groups = [[l for r, l in zip(rows, labels) if r[f] == v] for v in set(r[f] for r in rows)]
        gain = base - sum(len(g) / n * entropy(g) for g in groups)
        si = split_info(groups, n)
        return (gain / si if si > 0 else 0, gain, None)

    def scores(self, rows, labels, features):
        return {f: self._feature_score(rows, labels, f) for f in features}

    def _best_split(self, rows, labels, features):
        s = self.scores(rows, labels, features)
        # Quinlan's rule: only consider features with at least average gain,
        # then pick the highest gain ratio among them
        avg_gain = np.mean([v[1] for v in s.values()])
        candidates = {f: v for f, v in s.items() if v[1] >= avg_gain and v[1] > 0}
        if not candidates:
            return None
        best = max(candidates, key=lambda f: candidates[f][0])
        return best, candidates[best][2]

    def _make_node(self, split, rows, labels, features, depth):
        f, t = split
        if t is not None:            # numeric -> binary cut, feature can be reused
            return self._binary_node(f, "<=", t, rows, labels, features, depth)
        return self._multiway_node(f, rows, labels, features, depth)


# ============================================================
#                  CART  -  Gini Index, binary only
# ============================================================

class CART(BaseTree):
    def _candidate_splits(self, rows, f):
        if f in self.numeric:
            return [("<=", t) for t in self._thresholds([r[f] for r in rows])]
        return [("==", v) for v in sorted(set(r[f] for r in rows))]

    def scores(self, rows, labels, features):
        """Best (weighted_gini, op, value) for each feature. Lower is better."""
        n, out = len(labels), {}
        for f in features:
            best = (gini(labels), None, None)
            for op, v in self._candidate_splits(rows, f):
                L = [l for r, l in zip(rows, labels) if self._test(r[f], op, v)]
                R = [l for r, l in zip(rows, labels) if not self._test(r[f], op, v)]
                if not L or not R:
                    continue
                g = (len(L) * gini(L) + len(R) * gini(R)) / n
                if g < best[0]:
                    best = (g, op, v)
            out[f] = best
        return out

    def _best_split(self, rows, labels, features):
        s = self.scores(rows, labels, features)
        best = min(s, key=lambda f: s[f][0])
        return (best,) + s[best][1:] if s[best][1] is not None else None

    def _make_node(self, split, rows, labels, features, depth):
        f, op, v = split
        return self._binary_node(f, op, v, rows, labels, features, depth)


# ============================================================
#                  Helpers
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def discretize(train_col, test_col):
    """Turn numbers into low / mid / high using training-set thirds (for ID3)."""
    a, b = np.percentile(train_col, [33.3, 66.7])
    lab = lambda x: "low" if x <= a else ("mid" if x <= b else "high")
    return [lab(x) for x in train_col], [lab(x) for x in test_col]


def banner(text):
    print("\n" + "=" * 62 + f"\n{text}\n" + "=" * 62)


# ============================================================
#                  MAIN
# ============================================================

if __name__ == "__main__":
    # ------------------------------------------------------------
    # PART 1: Play Tennis (categorical) - all three algorithms
    # ------------------------------------------------------------
    cols = ["Outlook", "Temperature", "Humidity", "Wind"]
    raw = [
        ("Sunny", "Hot", "High", "Weak", "No"),      ("Sunny", "Hot", "High", "Strong", "No"),
        ("Overcast", "Hot", "High", "Weak", "Yes"),  ("Rain", "Mild", "High", "Weak", "Yes"),
        ("Rain", "Cool", "Normal", "Weak", "Yes"),   ("Rain", "Cool", "Normal", "Strong", "No"),
        ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
        ("Sunny", "Cool", "Normal", "Weak", "Yes"),  ("Rain", "Mild", "Normal", "Weak", "Yes"),
        ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
        ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No"),
    ]
    rows = [dict(zip(cols, r[:4])) for r in raw]
    labels = [r[4] for r in raw]

    banner("PART 1: Play Tennis - how each algorithm picks the ROOT question")
    print(f"Dataset: {len(rows)} days, {labels.count('Yes')} Yes / {labels.count('No')} No")
    print(f"Entropy = {entropy(labels):.3f}   Gini = {gini(labels):.3f}\n")

    id3, c45, cart = ID3(), C45(), CART()
    s_id3 = id3.scores(rows, labels, cols)
    s_c45 = c45.scores(rows, labels, cols)
    s_cart = cart.scores(rows, labels, cols)

    print(f"{'Feature':<13}{'ID3: Info Gain':>15}{'C4.5: Gain Ratio':>18}{'CART: Gini after':>18}")
    for f in cols:
        g, op, v = s_cart[f]
        print(f"{f:<13}{s_id3[f]:>15.3f}{s_c45[f][0]:>18.3f}{g:>12.3f} ({v})")
    print("\n(ID3 & C4.5 pick the HIGHEST score, CART picks the LOWEST Gini)")

    for name, model in (("ID3", id3), ("C4.5", c45), ("CART", cart)):
        model.fit(rows, labels, cols)
        print(f"\n--- {name} tree  (leaves: {model.count_leaves()}, "
              f"training accuracy: {accuracy(labels, model.predict(rows)):.2f}) ---")
        model.show()

    new_day = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
    print(f"\nNew day: {new_day}")
    for name, model in (("ID3", id3), ("C4.5", c45), ("CART", cart)):
        print(f"  {name:<5} says play? {model.predict_one(new_day)}")

    # ------------------------------------------------------------
    # PART 2: Iris (numeric) - all three algorithms
    # ------------------------------------------------------------
    banner("PART 2: Iris flowers (numeric features) - accuracy comparison")
    iris = load_iris()
    names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)
    y_tr_names = [iris.target_names[i] for i in y_tr]
    y_te_names = [iris.target_names[i] for i in y_te]

    num_tr = [dict(zip(names, x)) for x in X_tr]
    num_te = [dict(zip(names, x)) for x in X_te]

    # ID3 cannot handle numbers, so we first bin each feature into low/mid/high
    cat_tr = [{} for _ in X_tr]
    cat_te = [{} for _ in X_te]
    for j, n in enumerate(names):
        tr_bins, te_bins = discretize(X_tr[:, j], X_te[:, j])
        for row, b in zip(cat_tr, tr_bins):
            row[n] = b
        for row, b in zip(cat_te, te_bins):
            row[n] = b

    depth = 3
    id3_i = ID3(max_depth=depth).fit(cat_tr, y_tr_names, names)
    c45_i = C45(max_depth=depth, numeric=names).fit(num_tr, y_tr_names, names)
    cart_i = CART(max_depth=depth, numeric=names).fit(num_tr, y_tr_names, names)
    sk_acc = DecisionTreeClassifier(criterion="gini", max_depth=depth,
                                    random_state=0).fit(X_tr, y_tr).score(X_te, y_te)

    results = [
        ("ID3 (binned data)", accuracy(y_te_names, id3_i.predict(cat_te)), id3_i.count_leaves()),
        ("C4.5", accuracy(y_te_names, c45_i.predict(num_te)), c45_i.count_leaves()),
        ("CART", accuracy(y_te_names, cart_i.predict(num_te)), cart_i.count_leaves()),
        ("sklearn CART (check)", sk_acc, None),
    ]
    print(f"Train: {len(X_tr)} flowers, Test: {len(X_te)} flowers, max depth = {depth}\n")
    print(f"{'Algorithm':<22}{'Test accuracy':>14}{'Leaves':>9}")
    for name, acc, leaves in results:
        print(f"{name:<22}{acc:>14.4f}{(leaves if leaves else '-'):>9}")

    print("\n--- C4.5 tree on Iris ---")
    c45_i.show()
    print("\n--- CART tree on Iris ---")
    cart_i.show()

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    banner("SUMMARY")
    print(f"{'':<17}{'ID3':<18}{'C4.5':<20}{'CART'}")
    for row in [
        ("Split measure", "Information Gain", "Gain Ratio", "Gini Index"),
        ("Branches", "Multi-way", "Multi-way / binary", "Always binary"),
        ("Numeric data", "No (must bin)", "Yes (cut points)", "Yes (cut points)"),
        ("Task", "Classification", "Classification", "Classif. + Regression"),
        ("Bias", "Likes many-value", "Corrects ID3 bias", "Balanced splits"),
    ]:
        print(f"{row[0]:<17}{row[1]:<18}{row[2]:<20}{row[3]}")
