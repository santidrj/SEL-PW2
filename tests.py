import Cart as Cart
import pandas as pd
from src.classifiers import CART

df = pd.DataFrame({'eye-colour': ['blue', 'blue', 'brown', 'green', 'green', 'brown', 'green', 'blue'],
                   'hair-colour': ['blonde', 'brown', 'brown', 'brown', 'brown', 'brown', 'blone', 'brown'],
                   'height': ['tall', 'medium', 'medium', 'medium', 'tall', 'low', 'low', 'medium'],
                   'class': ['c+', 'c+', 'c-', 'c-', 'c+', 'c-', 'c-', 'c+']})

print(df)
cart = CART()
labels = df['class'].to_numpy()
cart._gini_index([df['eye-colour'].to_numpy()], labels)
