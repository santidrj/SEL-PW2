import logging
from math import ceil, log

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from src.classifiers import CART

# logging.basicConfig(level=logging.DEBUG)
from src.forest import DecisionForest, RandomForest

df = pd.DataFrame(
    {
        "eye-colour": [
            "blue",
            "blue",
            "brown",
            "green",
            "green",
            "brown",
            "green",
            "blue",
            "red",
            "red",
        ],
        "hair-colour": [
            "blonde",
            "brown",
            "brown",
            "green",
            "brown",
            "brown",
            "blonde",
            "brown",
            "green",
            "blonde",
        ],
        "height": [1.96, 1.75, 1.75, 1.63, 1.96, 1.45, 1.53, 1.69, 1.49, 1.85],
        "class": ["c+", "c+", "c-", "c-", "c+", "c-", "c-", "c+", "c-", "c+"],
    }
)

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)
# forest = RandomForest(2, 3)
forest = DecisionForest("Runif", 10)
forest.fit(x_train, y_train)
predictions = forest.predict(x_test)
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions, normalize="true")
plt.show()
print(forest.rule_count())
