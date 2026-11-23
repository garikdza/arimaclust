"""
The fitted Piccolo distance matrix of one curated set, as a 2-D MDS embedding (``figures/mds_embedding.png``)
and as a heatmap ordered by true cluster (``figures/distance_heatmap.png``).

The fit is saved to ``figures/distance_matrix.npz`` and later runs plot from it; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/distance_matrix.py``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering
from arimaclust.data.generate import generate_clusters
from results import load_or_compute, parse_args
from synthetic import CURATED_SETS

BAND = '0.2 ≤ δ < 0.4'
LENGTH = 400
N_PER_CLUSTER = 20
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'distance_matrix.npz'


def compute() -> dict[str, np.ndarray]:
    """Fit the clustering of the ``BAND`` set and return its distance matrix and labels."""
    curated = next(s for s in CURATED_SETS if s.band == BAND)
    np.random.seed(0)
    X, y = generate_clusters(curated.params, t_min=LENGTH, samples_per_cluster=N_PER_CLUSTER)
    model = ConsistentClustering(n_clusters=len(curated.params), distance=ARMADistance(n_jobs=-1)).fit(X)
    return {
        'distances': model.distance_.dist_matrix_,
        'y_true': np.asarray(y),
        'y_pred': model.labels_,
        'band': np.array(BAND),
        'delta': np.array(curated.delta),
        'length': np.array(LENGTH),
        'n_clusters': np.array(len(curated.params)),
        'n_per_cluster': np.array(N_PER_CLUSTER),
    }


def _save(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to ``FIGURES / name`` and close it."""
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def plot(results: dict[str, np.ndarray]) -> None:
    """Print the AMI and draw the MDS embedding and the heatmap of the saved distance matrix."""
    D, y, labels = results['distances'], results['y_true'], results['y_pred']
    band, delta, length = str(results['band']), float(results['delta']), int(results['length'])
    n_per_cluster = int(results['n_per_cluster'])
    ami = adjusted_mutual_info_score(y, labels)
    print(f'{band} (δ={delta}), T={length}: AMI {ami:.3f}')

    embedding = MDS(n_components=2, dissimilarity='precomputed', n_init=4, random_state=0).fit_transform(D)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, colours, title in zip(axes, (y, labels), ('true clusters', 'predicted clusters'), strict=True):
        ax.scatter(*embedding.T, c=colours, cmap='tab10', vmin=0, vmax=9)
        ax.set_title(title)
    fig.suptitle(f'MDS of the fitted Piccolo distances, {band} (δ={delta}), '
                 f'{n_per_cluster} series per cluster at T={length}: AMI {ami:.2f}')
    _save(fig, 'mds_embedding.png')

    # generate_clusters returns the series grouped by cluster, so the blocks are the clusters
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(D, cmap='viridis')
    fig.colorbar(image, ax=ax, label='Piccolo distance')
    for boundary in np.arange(1, int(results['n_clusters'])) * n_per_cluster - 0.5:
        ax.axhline(boundary, color='white', linewidth=0.8)
        ax.axvline(boundary, color='white', linewidth=0.8)
    ax.set_title(f'fitted distances by true cluster, {band} (δ={delta}), T={length}')
    ax.set_xlabel('series')
    ax.set_ylabel('series')
    _save(fig, 'distance_heatmap.png')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
