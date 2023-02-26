from arimaclust.models.centroid import ConsistentARMAClustering
import pandas as pd

df = pd.read_csv('data/final.csv', index_col=['Unnamed: 0'])
X = df.values
model = ConsistentARMAClustering(k=10, tau=100, max_orders=(2, 2))
model.fit(X=X)
print(model.labels_)
