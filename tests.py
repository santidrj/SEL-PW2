import logging

import pandas as pd

from sklearn.metrics import classification_report
from src.classifiers import CART

logging.basicConfig(level=logging.DEBUG)

df = pd.DataFrame({'eye-colour': ['blue', 'blue', 'brown', 'green', 'green', 'brown', 'green', 'blue', 'red', 'red'],
                   'hair-colour': ['blonde', 'brown', 'brown', 'green', 'brown', 'brown', 'blonde', 'brown', 'green',
                                   'blonde'],
                   'height': [1.96, 1.75, 1.75, 1.63, 1.96, 1.45, 1.53, 1.69, 1.49, 1.85],
                   'class': ['c+', 'c+', 'c-', 'c-', 'c+', 'c-', 'c-', 'c+', 'c-', 'c+']})

print(df)
cart = CART()
labels = df['class'].to_numpy()
cart.fit(df)
print(cart)
predictions = cart.predict(df.iloc[:, :-1])
print(classification_report(labels, predictions))


