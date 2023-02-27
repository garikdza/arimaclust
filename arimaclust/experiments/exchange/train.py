import numpy as np

from arimaclust.models.centroid import ConsistentARMAClustering
import pandas as pd

df = pd.read_csv('data/final.csv', index_col=['Unnamed: 0'])
print(df.index)
X = []
for col in df.columns:
    X.append(df[col].dropna())

# model = ConsistentARMAClustering(k=10, tau=100, max_orders=(2, 2))
# model.fit(X=X)
# np.save('labels.npy', model.labels_)
#
labels = np.load('labels.npy')
print(np.unique(labels, return_counts=True))
#
print(df.columns.tolist().index('AMD'))
print(labels[df.columns.tolist().index('AMD')])
print(df.columns[labels == 3])
print(labels)
