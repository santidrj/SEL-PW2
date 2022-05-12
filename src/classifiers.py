import logging
from itertools import chain, combinations
from typing import Tuple

import numpy as np
from pandas import DataFrame


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

class TreeNode:
    def __init__(self, data, is_leaf=False):
        self.left = None
        self.right = None
        self.data = data
        self.is_leaf = is_leaf


class CART:

    def __init__(self, max_depth=-1, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        self.classes = None
        self.columns = None
        self.numerical_columns = None
        self.logger = logging.getLogger('CART')

        self.validate_input()

    def __str__(self):
        s = "Preorder representation:\n"
        return s + self._print_tree(self.root)

    def validate_input(self):
        if self.max_depth == 0:
            raise ValueError("max_depth must be greater than 0 or -1 for no depth check.")
        if self.min_size < 1:
            raise ValueError("min_size must be greater than 0.")

    def fit(self, X):
        self.numerical_columns = X.select_dtypes(include='number').columns
        self.columns = X.iloc[:, :-1].columns
        self.classes = X.iloc[:, -1].unique()
        initial_split = self._get_split(X)
        self.root = TreeNode(initial_split)
        self._split(self.root, 1)

    def _split(self, node, depth):
        left = node.data['split'][0].copy()
        right = node.data['split'][1].copy()
        if len(left) < 1 or len(right) < 1:
            node.data = left.iloc[0, -1] if len(left) >= 1 else right.iloc[0, -1]
            node.is_leaf = True
            return

        del (node.data['split'])
        if 0 < self.max_depth <= depth:
            node.left = TreeNode(left.iloc[0, -1], True)
            node.right = TreeNode(right.iloc[0, -1], True)
            return

        if len(left) <= self.min_size:
            node.left = TreeNode(left.iloc[0, -1], True)
        else:
            left_split = self._get_split(left)
            node.left = TreeNode(left_split)
            self._split(node.left, depth + 1)

        if len(right) <= self.min_size:
            node.right = TreeNode(right.iloc[0, -1], True)
        else:
            right_split = self._get_split(right)
            node.right = TreeNode(right_split)
            self._split(node.right, depth + 1)

    def _get_split(self, X):
        best_gini, best_value, best_feature, best_split = np.Inf, None, None, None
        for c in self.columns:
            split, value, gini_idx = self._generate_splits(X, c)

            if gini_idx < best_gini:
                best_split = split
                best_gini = gini_idx
                best_feature = c
                best_value = value

            # exit if it finds a split with the best possible gini index.
            if best_gini == 0.0:
                break
        self.logger.info(f'Final split with gini index {best_gini} using feature {best_feature} and value/s {best_value}')
        return {'feature': best_feature, 'value': best_value, 'split': best_split}

    def _generate_splits(self, X, col):
        col_data = X[col]
        best_gini, best_value, best_split = np.Inf, None, None
        numerical = col in self.numerical_columns
        if numerical:
            col_values = sorted(col_data.unique())
        else:
            col_values = powerset(col_data.unique())

        for v in col_values:
            gini_idx, split = self._feature_splits(X, col_data, v, numerical)
            if gini_idx < best_gini:
                best_split = split
                best_gini = gini_idx
                best_value = v

            # exit if it finds a split with the best possible gini index.
            if best_gini == 0.0:
                break

        self.logger.info(f'Best split: {best_split} with gini index {best_gini}')
        return best_split, best_value, best_gini

    def _feature_splits(self, X, col_data, v, numerical):
        if numerical:
            mask = col_data <= v
        else:
            mask = col_data.isin(v)
        l_split = X[mask]
        r_split = X[~mask]
        split = (l_split, r_split)
        gini_idx = self._gini_index(split)
        return gini_idx, split

    def _gini_index(self, split: Tuple[DataFrame, DataFrame]) -> float:
        """
        Compute the Gini index for a given split. The value of the index lies between 0 and 1-1/|classes|.
        The smaller the value the greater separation.

        :param split: tuple representing a split
        :return: the gini index for the given split
        """
        gini = 0.0
        n_samples = sum([g.shape[0] for g in split])
        for g in split:
            if g.shape[0] > 0:
                score = 1.0
                # score the group based on each class
                for c in self.classes:
                    p = np.count_nonzero(g.iloc[:, -1] == c) / g.shape[0]
                    score -= p ** 2
                # weight the group score by size
                gini += (g.shape[0] / n_samples) * score
        return gini

    def _print_tree(self, node, depth=0):
        if node.is_leaf:
            return f'At level {depth} {node.data}\n'
        else:
            if node.data['feature'] in self.numerical_columns:
                s = f'At level {depth} [{node.data["feature"]} <= {node.data["value"]}?]\n'
            else:
                s = f'At level {depth} [{node.data["feature"]} is {node.data["value"]}?]\n'
            s += self._print_tree(node.left, depth + 1)
            s += self._print_tree(node.right, depth + 1)
            return s
