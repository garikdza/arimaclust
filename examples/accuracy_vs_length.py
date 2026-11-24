"""
Clustering accuracy against series length, one curve per curated δ band (``figures/accuracy_vs_length.png``).

The scores are saved to ``figures/accuracy_vs_length.npz`` and later runs plot from them; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/accuracy_vs_length.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from results import load_or_compute, parse_args
from style import BAND_COLOURS, MINIMAL, SURFACE
from synthetic import CURATED_SETS, clustering_scores

LENGTHS = (50, 100, 200, 400, 800, 1600)
N_PER_CLUSTER = 10
SEEDS = range(8)
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'accuracy_vs_length.npz'


def compute() -> dict[str, np.ndarray]:
    """Score the clustering of every curated set at every length over ``SEEDS``."""
    scores = []
    for curated in CURATED_SETS:
        print(f'fitting {curated.band} ...')
        scores.append([clustering_scores(curated.params, length, N_PER_CLUSTER, SEEDS) for length in LENGTHS])
    return {
        'bands': np.array([curated.band for curated in CURATED_SETS]),
        'deltas': np.array([curated.delta for curated in CURATED_SETS]),
        'lengths': np.array(LENGTHS),
        'scores': np.array(scores),
        'n_clusters': np.array(len(CURATED_SETS[0].params)),
        'n_per_cluster': np.array(N_PER_CLUSTER),
    }


def plot(results: dict[str, np.ndarray]) -> None:
    """Print the mean and spread of the saved scores and draw them against length, one curve per set."""
    lengths, scores = results['lengths'], results['scores']
    with plt.rc_context(MINIMAL):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for band, delta, set_scores, colour in zip(
            results['bands'], results['deltas'], scores, BAND_COLOURS, strict=True
        ):
            for length, length_scores in zip(lengths, set_scores, strict=True):
                print(f'{band:>14}  T={length:<5} AMI {length_scores.mean():.3f} ± {length_scores.std():.3f}')
            means, sds = set_scores.mean(axis=1), set_scores.std(axis=1)
            ax.fill_between(
                lengths, np.clip(means - sds, 0, 1), np.clip(means + sds, 0, 1), color=colour, alpha=0.08, linewidth=0
            )
            ax.plot(lengths, means, color=colour, label=f'{band}  (δ={float(delta)})')
            ax.scatter(lengths, means, s=30, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xscale('log')
        ax.set_xticks(lengths, [str(length) for length in lengths])
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.04)
        ax.set_xlabel('series length T')
        ax.set_ylabel(f'AMI (mean ± 1 sd, {scores.shape[-1]} draws)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='separation band')

        path = FIGURES / 'accuracy_vs_length.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
