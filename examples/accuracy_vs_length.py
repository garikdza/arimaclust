"""
Clustering accuracy against series length, one curve per curated δ band (``figures/accuracy_vs_length.png``).

Run with ``uv run --extra examples python examples/accuracy_vs_length.py``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthetic import CURATED_SETS, clustering_scores

LENGTHS = (50, 100, 200, 400, 800, 1600)
N_PER_CLUSTER = 10
SEEDS = range(8)
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curated in CURATED_SETS:
        scores = []
        for length in LENGTHS:
            scores.append(clustering_scores(curated.params, length, N_PER_CLUSTER, SEEDS))
            print(f'{curated.band:>14}  T={length:<5} AMI {scores[-1].mean():.3f} ± {scores[-1].std():.3f}')
        means, sds = np.mean(scores, axis=1), np.std(scores, axis=1)
        line, = ax.plot(LENGTHS, means, marker='o', label=f'{curated.band} (δ={curated.delta})')
        ax.fill_between(LENGTHS, means - sds, means + sds, color=line.get_color(), alpha=0.15)
    ax.set_xscale('log')
    ax.set_xticks(LENGTHS, [str(length) for length in LENGTHS])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('series length T')
    ax.set_ylabel(f'AMI (mean over {len(SEEDS)} draws)')
    ax.set_title(f'ConsistentClustering + ARMADistance (Piccolo, BIC), {len(CURATED_SETS[0].params)} clusters × '
                 f'{N_PER_CLUSTER} series')
    ax.legend(title='separation band')

    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / 'accuracy_vs_length.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
