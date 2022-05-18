from typing import List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample

from src.classifiers import CART


class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, n_trees, max_depth=-1, min_size=1, seed=None):
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.seed = seed
        self.rules_count = {}
        self.forest: List[CART] = [None] * self.n_trees
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        if self.n_features > X.shape[1]:
            raise ValueError(f"Cannot fit data with less than {self.n_features} features.")

        self.rules_count = {}

        for t in range(self.n_trees):
            x, target = resample(X, y, random_state=self.seed)
            tree = CART(
                max_depth=self.max_depth,
                min_size=self.min_size,
                n_features=self.n_features,
                random_state=self.rng,
            )
            tree.fit(x, target)
            self.forest[t] = tree

    def predict(self, X):
        votes = np.stack([t.predict(X) for t in self.forest], axis=1)
        preds = []
        for sample in votes:
            (v, c) = np.unique(sample, return_counts=True)
            i = np.argmax(c)
            preds.append(v[i])
        return np.array(preds)

    def feature_importance(self):
        if self.rules_count:
            return list(self.rules_count.keys())

        for t in self.forest:
            r = t.rule_count
            keys = list(r.keys())
            for f in keys:
                f_v = self.rules_count.get(f, {})
                r_v = r.get(f, {})
                self.rules_count[f] = {k: f_v.get(k, 0) + r_v.get(k, 0) for k in set(f_v) | set(r_v)}
        return list(self.rules_count.keys())


class DecisionForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features, n_trees, max_depth=-1, min_size=1, seed=None):
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.seed = seed
        self.rules_count = {}
        self.forest: List[CART] = [None] * self.n_trees
        self.tree_features: List[List[Union[int, str]]] = [None] * self.n_trees
        self.rng = np.random.default_rng(self.seed)

    def fit(self, X, y):
        if isinstance(self.n_features, int) and self.n_features > X.shape[1]:
            raise ValueError(f"Cannot fit data with less than {self.n_features} features.")

        self.rules_count = {}

        x = X.copy()
        target = y.copy()
        if not isinstance(X, DataFrame):
            x = pd.DataFrame(X)
        if not isinstance(y, Series):
            target = pd.Series(y, name="class")

        for t in range(self.n_trees):
            if isinstance(self.n_features, str) and self.n_features == "Runif":
                n_features = int(self.rng.uniform(1, X.shape[1] + 1))
                f_idx = self.rng.choice(x.columns, size=n_features, replace=False)
            else:
                f_idx = self.rng.choice(x.columns, size=self.n_features, replace=False)
            tree = CART(max_depth=self.max_depth, min_size=self.min_size, random_state=self.rng)
            tree.fit(x[f_idx], target)
            self.forest[t] = tree
            self.tree_features[t] = f_idx

    def predict(self, X):
        x = X.copy()
        if not isinstance(X, DataFrame):
            x = pd.DataFrame(X)

        votes = np.stack(
            [t.predict(x[self.tree_features[i]]) for i, t in enumerate(self.forest)],
            axis=1,
        )
        preds = []
        for sample in votes:
            (v, c) = np.unique(sample, return_counts=True)
            i = np.argmax(c)
            preds.append(v[i])
        return np.array(preds)

    def feature_importance(self):
        if self.rules_count:
            return list(self.rules_count.keys())

        for t in self.forest:
            r = t.rule_count
            keys = list(r.keys())
            for f in keys:
                f_v = self.rules_count.get(f, {})
                r_v = r.get(f, {})
                self.rules_count[f] = {k: f_v.get(k, 0) + r_v.get(k, 0) for k in set(f_v) | set(r_v)}
        return list(self.rules_count.keys())
