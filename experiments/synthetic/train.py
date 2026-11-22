import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust.data.examples import CLUSTER1
from arimaclust.data.generate import generate_clusters
from arimaclust.models.centroid import ConsistentARMAClustering

N_SAMPLES = 50
T_RANGE = [200, 400, 600, 800, 1000]
TAU = 10
KAPPA = len(CLUSTER1)

print(CLUSTER1)

scores = []
for t in T_RANGE:
    scores_per_t = []
    for j in range(10):
        print(f'================t_min:{t},run:{j}===============', '\n')
        X, Y = generate_clusters(params=CLUSTER1, t_min=t, samples_per_cluster=N_SAMPLES)
        model = ConsistentARMAClustering(k=KAPPA, metric='piccolo', metric_args={'tau': TAU},
                                         process_args={'max_p': 3, 'max_q': 3, 'd': 0})
        model.fit(X=X)
        y_pred = model.labels_
        score = adjusted_mutual_info_score(labels_true=Y, labels_pred=y_pred)
        scores_per_t.append(score)
    scores.append(scores_per_t)

m = np.mean(scores, axis=1)

e = np.std(scores, axis=1)

plt.errorbar(T_RANGE, m, e, linestyle='None', marker='^')
plt.savefig('t_vs_acc.png')
plt.show()
