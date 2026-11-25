"""
The fitted Piccolo distance matrix of one curated set (``figures/distance_matrix.png``): a heatmap ordered by true
cluster beside a 2-D MDS embedding coloured by true cluster, with the misclustered series circled.

The fit is saved to ``figures/distance_matrix.npz`` and later runs plot from it; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/synthetic/distance_matrix.py``.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering
from arimaclust.data import generate_clusters

# results.py and style.py are shared by every example and sit one folder up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from results import load_or_compute, parse_args
from style import CLUSTER_COLOURS, DISTANCE_RAMP, INK, MINIMAL, SURFACE
from synthetic import CURATED_SETS

BAND = '0.2 ≤ δ < 0.4'
LENGTH = 400
N_PER_CLUSTER = 20
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'distance_matrix.npz'


def compute() -> dict[str, np.ndarray]:
    """Fit the clustering of the ``BAND`` set and return its distance matrix and labels."""
    curated = next(s for s in CURATED_SETS if s.band == BAND)
    X, y = generate_clusters(curated.params, LENGTH, N_PER_CLUSTER, random_state=0)
    model = ConsistentClustering(n_clusters=len(curated.params), distance=ARMADistance(n_jobs=-1)).fit(X)
    return {
        'distances': model.distance_.dist_matrix_,
        'y_true': y,
        'y_pred': model.labels_,
        'band': np.array(BAND),
        'delta': np.array(curated.delta),
        'length': np.array(LENGTH),
        'n_clusters': np.array(len(curated.params)),
        'n_per_cluster': np.array(N_PER_CLUSTER),
    }


def plot(results: dict[str, np.ndarray]) -> None:
    """Print the AMI and draw the heatmap and the MDS embedding of the saved distance matrix side by side."""
    D, y, labels = results['distances'], results['y_true'], results['y_pred']
    band, delta, length = str(results['band']), float(results['delta']), int(results['length'])
    n_clusters, n_per_cluster = int(results['n_clusters']), int(results['n_per_cluster'])
    ami = adjusted_mutual_info_score(y, labels)
    print(f'{band} (δ={delta}), T={length}: AMI {ami:.3f}')
    cluster_names = [f'cluster {c + 1}' for c in range(n_clusters)]

    with plt.rc_context(MINIMAL):
        fig, (heat, mds) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={'width_ratios': [1.1, 1]})
        # generate_clusters returns the series grouped by cluster, so the blocks are the clusters
        heat.grid(False)
        image = heat.imshow(
            D, cmap=LinearSegmentedColormap.from_list('distance', DISTANCE_RAMP), interpolation='nearest'
        )
        for boundary in np.arange(1, n_clusters) * n_per_cluster - 0.5:
            heat.axhline(boundary, color=SURFACE, linewidth=2)
            heat.axvline(boundary, color=SURFACE, linewidth=2)
        centres = np.arange(n_clusters) * n_per_cluster + n_per_cluster / 2 - 0.5
        heat.set_xticks(centres, cluster_names)
        heat.set_yticks(centres, cluster_names)
        heat.spines[['left', 'bottom']].set_visible(False)
        bar = fig.colorbar(image, ax=heat, fraction=0.046, pad=0.03)
        bar.outline.set_visible(False)
        bar.set_label('Piccolo distance (lighter = closer)')

        embedding = MDS(n_components=2, dissimilarity='precomputed', n_init=4, random_state=0).fit_transform(D)
        # a predicted cluster stands for the true cluster most of its series come from; the rest are misclustered
        majority = {label: np.bincount(y[labels == label]).argmax() for label in np.unique(labels)}
        wrong = np.array([majority[label] != true for label, true in zip(labels, y, strict=True)])
        mds.grid(False)
        for c, colour in zip(range(n_clusters), CLUSTER_COLOURS, strict=True):
            mds.scatter(
                *embedding[y == c].T,
                s=40,
                color=colour,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=cluster_names[c],
                zorder=3,
            )
        mds.scatter(
            *embedding[wrong].T, s=130, facecolor='none', edgecolor=INK, linewidth=1, zorder=4, label='misclustered'
        )
        mds.set_xticks([])
        mds.set_yticks([])
        mds.spines[['left', 'bottom']].set_visible(False)
        mds.legend(loc='best', title=f'true cluster (AMI {ami:.2f})')

        path = FIGURES / 'distance_matrix.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
