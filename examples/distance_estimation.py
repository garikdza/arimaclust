"""
The Piccolo distance between two fitted models against series length, for the closest pair of each curated set
(``figures/estimated_distance.png``). The spread of the estimates shrinks as the series grow and their mean
approaches the true distance δ. At these lengths BIC still often picks lower orders than the true ones, most for
the closest set, where an AR(1) fits both the ARMA(1,1) and the MA(1) process well; that pulls its estimate below δ.

The estimates are saved to ``figures/distance_estimation.npz`` and later runs plot from them; ``--recompute`` fits
again.

Run with ``uv run --extra examples python examples/distance_estimation.py``.
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arimaclust import ARMADistance
from arimaclust.data.generate import generate_clusters
from results import load_or_compute, parse_args
from style import BAND_COLOURS, INK_SOFT, MINIMAL
from synthetic import CURATED_SETS, min_separation

LENGTHS = (50, 100, 200, 400, 800, 1600)
N_PAIRS = 20
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'distance_estimation.npz'
LINE = BAND_COLOURS[2]


def compute() -> dict[str, np.ndarray]:
    """Estimate the distance of each curated set's closest pair ``N_PAIRS`` times per length."""
    pairs, estimates = [], []
    for curated in CURATED_SETS:
        print(f'fitting {curated.band} ...')
        a, b = min(itertools.combinations(curated.params, 2), key=lambda pair: min_separation(list(pair)))
        pairs.append(f'{a} vs {b}')
        set_estimates = []
        for length in LENGTHS:
            np.random.seed(length)
            X, _ = generate_clusters([a, b], t_min=length, samples_per_cluster=N_PAIRS)
            # one fit of all series; series k of the first process is paired with series k of the second
            D = ARMADistance(n_jobs=-1).fit_transform(X)
            set_estimates.append(D[np.arange(N_PAIRS), N_PAIRS + np.arange(N_PAIRS)])
        estimates.append(set_estimates)
    return {
        'bands': np.array([curated.band for curated in CURATED_SETS]),
        'deltas': np.array([curated.delta for curated in CURATED_SETS]),
        'pairs': np.array(pairs),
        'lengths': np.array(LENGTHS),
        'estimates': np.array(estimates),
    }


def plot(results: dict[str, np.ndarray]) -> None:
    """Print the mean and spread of the saved estimates and draw them against length, one panel per set."""
    lengths, estimates = results['lengths'], results['estimates']
    with plt.rc_context(MINIMAL):
        fig, axes = plt.subplots(1, len(estimates), figsize=(15, 3.6), gridspec_kw={'wspace': 0.6})
        for ax, band, delta, pair, set_estimates in zip(axes, results['bands'], results['deltas'], results['pairs'],
                                                        estimates, strict=True):
            delta = float(delta)  # a plain float prints as before, e.g. 0.801
            print(f'{band:>14}  closest pair {pair}')
            for length, length_estimates in zip(lengths, set_estimates, strict=True):
                print(f'{band:>14}  T={length:<5} estimate {length_estimates.mean():.3f} ± '
                      f'{length_estimates.std():.3f}  (δ={delta})')
            means, sds = set_estimates.mean(axis=1), set_estimates.std(axis=1)
            ax.axhline(delta, color=INK_SOFT, linewidth=1)
            ax.annotate(f'true δ = {delta}', (1.0, delta), xycoords=('axes fraction', 'data'), xytext=(4, 0),
                        textcoords='offset points', ha='left', va='center', fontsize=7.5, color=INK_SOFT,
                        annotation_clip=False)
            ax.fill_between(lengths, means - sds, means + sds, color=LINE, alpha=0.12, linewidth=0)
            ax.plot(lengths, means, color=LINE)
            ax.set_xscale('log')
            ax.set_xticks(lengths, [str(length) for length in lengths], fontsize=7.5)
            ax.minorticks_off()
            ax.set_title(band, fontsize=10, pad=6)
            ax.set_xlabel('series length T')
        axes[0].set_ylabel(f'Piccolo distance (mean ± 1 sd, {estimates.shape[-1]} fits)')

        path = FIGURES / 'estimated_distance.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
