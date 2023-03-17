import numpy as np
from scipy.optimize import linear_sum_assignment


def mapp_hungarian(labels1, labels2):
    # Define the number of clusters for each time period
    n_c1, n_c2 = int(max(labels1))+1, int(max(labels2))+1
    # Compute the confusion matrix between the two sets of labels
    confusion_matrix = np.zeros((n_c1, n_c2))

    for i in range(n_c1):
        for j in range(n_c2):
            confusion_matrix[i, j] = np.sum((labels1 == i) & (labels2 == j))

    print(confusion_matrix)
    # Use the Hungarian algorithm to find the optimal mapping of clusters
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    return row_ind, col_ind


def min_weight_bipartite_matching(clustering_a, clustering_b):
    ids_a = np.unique(clustering_a)  # distinct cluster ids in a
    ids_b = np.unique(clustering_b)  # distinct cluster ids in b
    n_a = len(clustering_a)  # number of instances in a
    n_b = len(clustering_b)  # number of instances in b
    if len(ids_a) != len(ids_b) or n_a != n_b:
        raise ValueError("number of cluster or number of instances do not match")

    n_c = len(ids_a)
    tupel = np.arange(n_a) + 1

    # computing the distance matrix
    assignment_matrix = np.full((n_c, n_c), -1)
    for i in range(n_c):
        tupel_cluster_i = tupel[clustering_a == ids_a[i]]
        sol_row_i = np.array(
            [np.sum(np.logical_and(clustering_b == ids_b[j], np.isin(tupel, tupel_cluster_i, assume_unique=True))) for j
             in range(n_c)])
        sol_row_i = (len(tupel_cluster_i) - sol_row_i) + (np.sum(clustering_b == ids_b[i]) - sol_row_i)
        assignment_matrix[i, :] = sol_row_i

    # optimization
    row_ind, col_ind = linear_sum_assignment(assignment_matrix, maximize=False)
    result = (row_ind, col_ind)
    result.assignment_matrix = assignment_matrix
    return result


if __name__ == '__main__':
    from arimaclust.data.examples import CLUSTER1
    from arimaclust.data.generate import generate_clusters
    from arimaclust.models.centroid import ConsistentARMAClustering
    from arimaclust.models.neighbours import NearestNeighbourModel

    N_SAMPLES = 50
    T_RANGE = 400
    TAU = 10
    KAPPA = len(CLUSTER1)
    metric_args = {'tau': TAU}
    process_args = {'max_p': 3, 'max_q': 3, 'd': 0}

    print(CLUSTER1)

    scores = []
    X, Y = generate_clusters(params=CLUSTER1, t_min=T_RANGE, samples_per_cluster=N_SAMPLES)

    X1 = [x[:200] for x in X]
    model = NearestNeighbourModel(alpha=0.1, metric='piccolo', metric_args=metric_args, process_args=process_args)
    model.fit(X=X1)
    y_pred1 = model.labels_

    X2 = [x[200:] for x in X]

    model = NearestNeighbourModel(alpha=0.1, metric='piccolo', metric_args=metric_args, process_args=process_args)
    model.fit(X=X2)
    y_pred2 = model.labels_

    match = mapp_hungarian(labels1=y_pred1, labels2=y_pred2)

    print(y_pred1)
    print(y_pred2)

    print(match)
