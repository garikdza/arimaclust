"""
The fitted Piccolo distance matrix of one curated set, as a 2-D MDS embedding (``figures/mds_embedding.png``)
and as a heatmap ordered by true cluster (``figures/distance_heatmap.png``).

Run with ``uv run --extra examples python examples/distance_matrix.py``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering
from arimaclust.data.generate import generate_clusters
from synthetic import CURATED_SETS

BAND = '0.2 ≤ δ < 0.4'
LENGTH = 400
N_PER_CLUSTER = 20
FIGURES = Path(__file__).parent / 'figures'


def _save(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to ``FIGURES / name`` and close it."""
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    curated = next(s for s in CURATED_SETS if s.band == BAND)
    np.random.seed(0)
    X, y = generate_clusters(curated.params, t_min=LENGTH, samples_per_cluster=N_PER_CLUSTER)
    model = ConsistentClustering(n_clusters=len(curated.params), distance=ARMADistance(n_jobs=-1)).fit(X)
    D = model.distance_.dist_matrix_
    ami = adjusted_mutual_info_score(y, model.labels_)
    print(f'{BAND} (δ={curated.delta}), T={LENGTH}: AMI {ami:.3f}')
    FIGURES.mkdir(exist_ok=True)

    embedding = MDS(n_components=2, dissimilarity='precomputed', n_init=4, random_state=0).fit_transform(D)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, labels, title in zip(axes, (y, model.labels_), ('true clusters', 'predicted clusters'), strict=True):
        ax.scatter(*embedding.T, c=labels, cmap='tab10', vmin=0, vmax=9)
        ax.set_title(title)
    fig.suptitle(f'MDS of the fitted Piccolo distances, {BAND} (δ={curated.delta}), '
                 f'{N_PER_CLUSTER} series per cluster at T={LENGTH}: AMI {ami:.2f}')
    _save(fig, 'mds_embedding.png')

    # generate_clusters returns the series grouped by cluster, so the blocks are the clusters
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(D, cmap='viridis')
    fig.colorbar(image, ax=ax, label='Piccolo distance')
    for boundary in np.arange(1, len(curated.params)) * N_PER_CLUSTER - 0.5:
        ax.axhline(boundary, color='white', linewidth=0.8)
        ax.axvline(boundary, color='white', linewidth=0.8)
    ax.set_title(f'fitted distances by true cluster, {BAND} (δ={curated.delta}), T={LENGTH}')
    ax.set_xlabel('series')
    ax.set_ylabel('series')
    _save(fig, 'distance_heatmap.png')


if __name__ == '__main__':
    main()
