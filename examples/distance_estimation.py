"""
The Piccolo distance between two fitted models against series length, for the closest pair of each curated set
(``figures/estimated_distance.png``). The spread of the estimates shrinks as the series grow and their mean
approaches the true distance δ. At these lengths BIC still often picks lower orders than the true ones, most for
the closest set, where an AR(1) fits both the ARMA(1,1) and the MA(1) process well; that pulls its estimate below δ.

Run with ``uv run --extra examples python examples/distance_estimation.py``.
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arimaclust import ARMADistance
from arimaclust.data.generate import generate_clusters
from synthetic import CURATED_SETS, min_separation

LENGTHS = (50, 100, 200, 400, 800, 1600)
N_PAIRS = 20
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, curated in zip(axes.flat, CURATED_SETS, strict=True):
        a, b = min(itertools.combinations(curated.params, 2), key=lambda pair: min_separation(list(pair)))
        estimates = []
        for length in LENGTHS:
            np.random.seed(length)
            X, _ = generate_clusters([a, b], t_min=length, samples_per_cluster=N_PAIRS)
            # one fit of all series; series k of the first process is paired with series k of the second
            D = ARMADistance(n_jobs=-1).fit_transform(X)
            estimates.append(D[np.arange(N_PAIRS), N_PAIRS + np.arange(N_PAIRS)])
            print(f'{curated.band:>14}  T={length:<5} estimate {estimates[-1].mean():.3f} ± {estimates[-1].std():.3f}'
                  f'  (δ={curated.delta})')
        means, sds = np.mean(estimates, axis=1), np.std(estimates, axis=1)
        ax.plot(LENGTHS, means, marker='o', label=f'mean of {N_PAIRS} estimates')
        ax.fill_between(LENGTHS, means - sds, means + sds, alpha=0.2, label='± 1 sd')
        ax.axhline(curated.delta, color='black', linestyle='--', label=f'true δ = {curated.delta}')
        ax.set_title(f'{curated.band}: {a} vs {b}', fontsize=9)
        ax.set_ylabel('Piccolo distance')
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xscale('log')
        ax.set_xticks(LENGTHS, [str(length) for length in LENGTHS])
        ax.set_xlabel('series length T')
    fig.suptitle('Piccolo distance between fitted models (ARMADistance, BIC) for the closest pair of each set')

    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / 'estimated_distance.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
