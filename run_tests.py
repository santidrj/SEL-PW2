import os.path
from math import log, sqrt

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold

from src.datasets import load_iris
from src.forest import RandomForest

output_dir = 'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# NT = (1, 10, 25, 50, 75, 100)
NT = (1, 10)


def test_random_forest(dataset_name, n_splits=5, n_jobs=None, seed=None):
    if dataset_name == 'iris':
        X, y = load_iris()
    else:
        raise ValueError(f'There is no dataset available with name {dataset_name}')

    f = {1, 3, int(log(X.shape[1], 2) + 1), int(sqrt(X.shape[1]))}

    results = pd.DataFrame(index=f, columns=NT)
    output_lines = np.empty((len(NT) * len(f),), dtype=object)
    for i, nt in enumerate(NT):
        for j, n_features in enumerate(f):
            forest = RandomForest(n_features, nt)
            cv_results = cross_validate(forest, X, y, scoring='accuracy',
                                        cv=StratifiedKFold(n_splits, random_state=seed), n_jobs=n_jobs,
                                        return_estimator=True)

            acc = cv_results['test_score']
            estimator = cv_results['estimator'][np.argmax(acc)]
            output_lines[i * len(f) + j] = f'Feature relevance for NT={nt}, F={n_features}: {estimator.rule_count()}\n'

            results[nt].loc[n_features] = np.mean(cv_results['test_score'])

    with open(os.path.join(output_dir, f'RandomForest-{dataset_name}-feature-relevance.txt'), 'w') as f:
        f.writelines(output_lines)

    print(results)
    return results


if __name__ == '__main__':
    test_random_forest('iris', n_jobs=-1)
