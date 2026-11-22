import numpy as np
from scipy.optimize import linear_sum_assignment


def contingency_matrix(labels1, labels2) -> np.ndarray:
    """Counts of samples with label ``i`` in ``labels1`` and label ``j`` in ``labels2``."""
    labels1, labels2 = np.asarray(labels1, dtype=int), np.asarray(labels2, dtype=int)
    n_c1, n_c2 = labels1.max() + 1, labels2.max() + 1
    matrix = np.zeros((n_c1, n_c2))
    np.add.at(matrix, (labels1, labels2), 1)
    return matrix


def mapp_hungarian(labels1, labels2):
    """
    Match clusters of two labelings of the same samples (e.g. consecutive time windows)
    by maximising overlap with the Hungarian algorithm. Returns ``(row_ind, col_ind)``:
    cluster ``row_ind[i]`` in ``labels1`` maps to cluster ``col_ind[i]`` in ``labels2``.
    """
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix(labels1, labels2))
    return row_ind, col_ind
