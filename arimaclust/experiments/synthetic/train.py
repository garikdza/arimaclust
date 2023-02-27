import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score
from arimaclust.data.generate import generate_clusters
from arimaclust.data.examples import EXAMPLE1, EXAMPLE2, CLUSTER1
from arimaclust.models.centroid import ConsistentARMAClustering

N_SAMPLES = 50
T_RANGE = [200, 400, 600, 800, 1000]
TAU = 100
KAPPA = len(EXAMPLE1)

scores = []
for t in T_RANGE:
    print(f'================{t}===============', '\n')
    X, Y = generate_clusters(params=CLUSTER1, t_min=t, samples_per_cluster=N_SAMPLES)
    model = ConsistentARMAClustering(k=KAPPA, metric='piccolo', tau=TAU, max_orders=(3, 3))
    model.fit(X=X)
    y_pred = model.labels_
    score = adjusted_mutual_info_score(labels_true=Y, labels_pred=y_pred)
    scores.append(score)

plt.plot(T_RANGE, scores)
plt.savefig('t_vs_acc.png')
plt.show()
