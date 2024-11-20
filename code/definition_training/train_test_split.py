from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('datasets/train.csv').sample(20)
a, b = train_test_split(df, test_size=0.1)

a.to_csv('datasets/train_def.csv', index=False)
b.to_csv('datasets/val_def.csv', index=False)
