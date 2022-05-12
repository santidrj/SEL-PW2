import logging

import pandas as pd

from src.classifiers import CART

logging.basicConfig(level=logging.DEBUG)

df = pd.DataFrame({'eye-colour': ['blue', 'blue', 'brown', 'green', 'green', 'brown', 'green', 'blue'],
                   'hair-colour': ['blonde', 'brown', 'brown', 'brown', 'brown', 'brown', 'blone', 'brown'],
                   'height': [1.96, 1.75, 1.75, 1.63, 1.96, 1.45, 1.53, 1.69],
                   'class': ['c+', 'c+', 'c-', 'c-', 'c+', 'c-', 'c-', 'c+']})

print(df)
cart = CART()
labels = df['class'].to_numpy()
cart.fit(df)
