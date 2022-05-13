from typing import List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.utils import resample

from src.classifiers import CART


class RandomForest:
    def __init__(self, n_features, n_trees, n_bootstrap, max_depth=-1, min_size=1, seed=None):
        self.n_features = n_features
        self.n_trees = n_trees
        self.n_bootstrap = n_bootstrap
        self.max_depth = max_depth
        self.min_size = min_size
        self.rule_count = {}
        self.forest: List[CART] = [None] * self.n_trees
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.rule_count = {}

        for t in range(self.n_trees):
            X_b, y_b = resample(X, y, n_samples=self.n_bootstrap, replace=False)
            tree = CART(max_depth=self.max_depth, min_size=self.min_size, n_features=self.n_features,
                        random_state=self.rng)
            tree.fit(X_b, y_b)
            self.forest[t] = tree

    def predict(self, X):
        votes = np.stack([t.predict(X) for t in self.forest], axis=1)
        preds = []
        for sample in votes:
            (v, c) = np.unique(sample, return_counts=True)
            i = np.argmax(c)
            preds.append(v[i])
        return np.array(preds)

    def rule_count(self):
        if self.rule_count:
            return self.rule_count

        for t in self.forest:
            r = t.rule_count()
            keys = list(r.keys())
            for f in keys:
                f_v = self.rule_count.get(f, {})
                r_v = r.get(f, {})
                self.rule_count[f] = {k: f_v.get(k, 0) + r_v.get(k, 0) for k in set(f_v) | set(r_v)}
        return self.rule_count


class DecisionForest:
    def __init__(self, n_features, n_trees, max_depth=-1, min_size=1, seed=None):
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.rule_count = {}
        self.forest: List[CART] = [None] * self.n_trees
        self.tree_features: List[List[Union[int, str]]] = [None] * self.n_trees
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        self.rule_count = {}

        x = X.copy()
        target = y.copy()
        if not isinstance(X, DataFrame):
            x = pd.DataFrame(X)
        if not isinstance(y, Series):
            target = pd.Series(y, name='class')

        for t in range(self.n_trees):
            f_idx = self.rng.choice(x.columns, size=self.n_features, replace=False)
            tree = CART(max_depth=self.max_depth, min_size=self.min_size, random_state=self.rng)
            tree.fit(x[f_idx], target)
            self.forest[t] = tree
            self.tree_features[t] = f_idx

    def predict(self, X):
        x = X.copy()
        if not isinstance(X, DataFrame):
            x = pd.DataFrame(X)

        votes = np.stack([t.predict(x[self.tree_features[i]]) for i, t in enumerate(self.forest)], axis=1)
        preds = []
        for sample in votes:
            (v, c) = np.unique(sample, return_counts=True)
            i = np.argmax(c)
            preds.append(v[i])
        return np.array(preds)

    def rule_count(self):
        if self.rule_count:
            return self.rule_count

        for t in self.forest:
            r = t.rule_count()
            keys = list(r.keys())
            for f in keys:
                f_v = self.rule_count.get(f, {})
                r_v = r.get(f, {})
                self.rule_count[f] = {k: f_v.get(k, 0) + r_v.get(k, 0) for k in set(f_v) | set(r_v)}
        return self.rule_count
