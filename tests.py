from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from src.datasets import load_nursery
from src.forest import RandomForest

# logging.basicConfig(level=logging.DEBUG)

rice, labels = load_nursery()
x_train, x_test, y_train, y_test = train_test_split(rice, labels, test_size=0.2)
forest = RandomForest(1, 1)
# forest = DecisionForest("Runif", 10)
forest.fit(x_train, y_train)
predictions = forest.predict(x_test)
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions, normalize="true")
plt.show()
print(forest.feature_importance())
