import os.path
from argparse import ArgumentParser
from math import log, sqrt
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.datasets import load_dataset, load_heart, load_iris, load_rice
from src.forest import DecisionForest, RandomForest


def test_random_forest(
    dataset_name, classifier_name, n_splits=5, n_jobs=None, seed=None, verbose=False
) -> Tuple[DataFrame, DataFrame]:
    """
    Perform cross-validation with a given dataset and classifier with a predefined number of combinations of number of
    trees and features.

    The output files will be stored in the "output" directory.

    Parameters
    ----------
    dataset_name : str
        The dataset name of one of the available datasets or the relative path to a dataset in CSV
        format. The available dataset names are iris, heart, and rice. If a dataset file is used, the functions assumes
        that the class labels are at the last column.
    classifier_name : str
        The classifier to use. Either RandomForest or DecisionForest.
    n_splits :
        The number of splits for the `StratifiedKFold` used for cross-validation.
    n_jobs :
        Number of jobs to run in parallel. ``-1`` means using all processors.
    seed :
        Controls the pseudo number generator.
    verbose:
        If true print the progress of the test.

    Returns
    -------
    results : DataFrame
        The results of the cross-validation for each combination of NT and F.
    feature_importance : DataFrame
        The feature importance for each combination of NT and F.
    """

    if dataset_name == "iris":
        X, y = load_iris()
    elif dataset_name == "heart":
        X, y = load_heart()
    elif dataset_name == "rice":
        X, y = load_rice()
    else:
        X, y = load_dataset(dataset_name)
        dataset_name = os.path.basename(dataset_name).split(".")[0]

    if classifier_name not in ["RandomForest", "DecisionForest"]:
        raise ValueError(f"Unknown classifier {classifier_name}")

    if classifier_name == "RandomForest":
        f = {1, 3, int(log(X.shape[1], 2) + 1), int(sqrt(X.shape[1]))}
    else:
        m = X.shape[1]
        f = {m // 4, m // 2, 3 * m // 4, "Runif"}

    if verbose:
        print(f"\nStarting tests for {classifier_name} with dataset {dataset_name}")
    if classifier_name == "RandomForest":
        return run_cross_validation(
            X, y, RandomForest, dataset_name, f, n_jobs, n_splits, seed, verbose
        )
    else:
        return run_cross_validation(
            X, y, DecisionForest, dataset_name, f, n_jobs, n_splits, seed, verbose
        )


def run_cross_validation(
    X, y, classifier, dataset_name, f, n_jobs, n_splits, seed, verbose
):
    results = pd.DataFrame(index=f, columns=NT)
    results = results.rename_axis(index="F")
    index = pd.MultiIndex.from_product([NT, f], names=["NT", "F"])
    col_names = [f"#{i + 1}" for i in range(X.shape[1])]
    feature_importance = pd.DataFrame(index=index, columns=col_names, dtype=str)
    output_lines = np.empty((len(NT) * len(f),), dtype=object)
    for i, nt in enumerate(NT):
        for j, n_features in enumerate(f):
            if verbose:
                print(f"Starting cross validation for NT={nt} and F={n_features}")

            forest = classifier(n_features, nt)
            cv_results = cross_validate(
                forest,
                X,
                y,
                scoring="accuracy",
                cv=StratifiedKFold(n_splits, random_state=seed),
                n_jobs=n_jobs,
                return_estimator=True,
            )

            acc = cv_results["test_score"]
            estimator = cv_results["estimator"][np.argmax(acc)]
            features = np.full(X.shape[1], "-", dtype=object)
            rules = estimator.rule_count()
            features[: len(rules)] = rules
            feature_importance.loc[(nt, n_features)] = features
            output_lines[
                i * len(f) + j
            ] = f"Feature relevance for NT={nt}, F={n_features}: {estimator.rule_count()}\n"

            results.loc[n_features, nt] = np.mean(cv_results["test_score"])

    results.columns = pd.MultiIndex.from_tuples(
        map(lambda x: ("NT", x), results.columns)
    )
    results = results.round(decimals=4)
    feature_importance.columns = pd.MultiIndex.from_tuples(
        map(lambda x: ("Feature importance", x), feature_importance.columns)
    )
    output_file = os.path.join(OUTPUT_DIR, f"{classifier.__name__}-{dataset_name}")

    with open(
        os.path.join(OUTPUT_DIR, f"{output_file}-feature-relevance.txt"), "w"
    ) as f:
        f.writelines(output_lines)

    results.to_pickle(f"{output_file}-results.pkl")
    feature_importance.to_pickle(f"{output_file}-feature-relevance.pkl")

    feature_importance.style.to_latex(
        f"{output_file}-feature-relevance.tex",
        position="htbp",
        position_float="centering",
        hrules=True,
        multicol_align="c",
        label=f"{dataset_name}-features",
    )
    s = results.style.highlight_max(props="textbf: --rwrap", axis=0)
    s.to_latex(
        f"{output_file}-results.tex",
        position="htbp",
        position_float="centering",
        hrules=True,
        multicol_align="c",
        label=f"{dataset_name}-results",
    )
    return results, feature_importance


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "dataset",
        help="The dataset to use for testing. Available datasets are: iris, heart, rice, all. You can also pass the "
        "relative path to a dataset in CSV format.",
    )
    parser.add_argument(
        "classifier",
        choices=("RandomForest", "DecisionForest", "all"),
        help="The classifier to test.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        help="The number of jobs to run in parallel. It defaults to -1 to use all the available CPUs. Defaults to -1.",
        default=-1,
    )
    parser.add_argument(
        "-n",
        "--n_splits",
        help="The number of splits for the StratifiedKFold. Defaults to 5.",
        default=5,
    )
    parser.add_argument("-s", "--seed", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    DATASET = args.dataset
    CLASSIFIER = args.classifier
    N_JOBS = args.jobs
    N_SPLITS = args.n_splits
    SEED = args.seed
    VERBOSE = args.verbose

    OUTPUT_DIR = "output"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    NT = (1, 10, 25, 50, 75, 100)

    if DATASET == "all":
        datasets = ("iris", "heart", "rice")
    else:
        datasets = tuple(DATASET)

    if CLASSIFIER == "all":
        classifiers = ("RandomForest", "DecisionForest")
    else:
        classifiers = tuple(CLASSIFIER)

    for dataset in datasets:
        for classifier in classifiers:
            test_random_forest(dataset, classifier, N_SPLITS, N_JOBS, SEED, VERBOSE)
