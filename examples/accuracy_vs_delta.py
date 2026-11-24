"""
Clustering accuracy against separation δ on generated two-cluster sets (``figures/accuracy_vs_delta.png``).

Each band gets sets from ``generate_separated_params`` whose δ falls inside it. The spread within a band is wide,
because δ is not the only thing that makes a pair of processes hard to tell apart from short series.

The scores are saved to ``figures/accuracy_vs_delta.npz`` and later runs plot from them; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/accuracy_vs_delta.py``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from results import load_or_compute, parse_args
from style import AXIS, BAND_COLOURS, MINIMAL, SURFACE
from synthetic import BANDS, TAU, band_of, clustering_scores, generate_separated_params, min_separation

LENGTH = 200
N_PER_CLUSTER = 10
SEEDS = range(4)
SETS_PER_BAND = 6
# with two clusters δ is the only pairwise distance; a far third cluster would lift every band's accuracy
N_CLUSTERS = 2
DELTA_TICKS = (0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5)
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'accuracy_vs_delta.npz'


def compute() -> dict[str, np.ndarray]:
    """Generate ``SETS_PER_BAND`` sets per band and score the clustering of each over ``SEEDS``."""
    bands, seeds, deltas, scores = [], [], [], []
    for i, (label, (low, _)) in enumerate(BANDS.items()):
        print(f'fitting {label} ...')
        found = 0
        seed = 1000 * i
        while found < SETS_PER_BAND:
            params = generate_separated_params(N_CLUSTERS, low, seed)
            delta = min_separation(params)
            if band_of(delta) == label:
                bands.append(label)
                seeds.append(seed)
                deltas.append(delta)
                scores.append(clustering_scores(params, LENGTH, N_PER_CLUSTER, SEEDS))
                found += 1
            seed += 1
    return {
        'bands': np.array(bands),
        'seeds': np.array(seeds),
        'deltas': np.array(deltas),
        'scores': np.array(scores),
        'length': np.array(LENGTH),
        'n_clusters': np.array(N_CLUSTERS),
        'n_per_cluster': np.array(N_PER_CLUSTER),
    }


def plot(results: dict[str, np.ndarray]) -> None:
    """Print each set's mean AMI and each band's mean, and draw them against δ."""
    bands, seeds, deltas = results['bands'], results['seeds'], results['deltas']
    set_scores = results['scores'].mean(axis=1)
    with plt.rc_context(MINIMAL):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for label, colour in zip(BANDS, BAND_COLOURS, strict=True):
            in_band = bands == label
            for seed, delta, score in zip(seeds[in_band], deltas[in_band], set_scores[in_band], strict=True):
                print(f'{label:>14}  seed={seed:<5} δ={delta:.4f}  AMI {score:.3f}')
            mean = set_scores[in_band].mean()
            print(f'{label:>14}  mean AMI {mean:.3f}')
            ax.scatter(deltas[in_band], set_scores[in_band], s=42, color=colour, edgecolor=SURFACE, linewidth=2,
                       zorder=3, label=f'{label}   mean {mean:.2f}')
            ax.hlines(mean, deltas[in_band].min(), deltas[in_band].max(), color=colour, linewidth=2.5, zorder=2)
        for boundary, _ in list(BANDS.values())[1:]:
            ax.axvline(boundary, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_xscale('log')
        ticks = [x for x in DELTA_TICKS if deltas.min() <= x <= deltas.max()]
        ax.set_xticks(ticks, [f'{x:g}' for x in ticks])
        ax.minorticks_off()
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel(f'separation δ (smallest Piccolo distance, τ = {TAU})')
        ax.set_ylabel(f'AMI (mean of {results["scores"].shape[-1]} draws)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title='separation band')

        path = FIGURES / 'accuracy_vs_delta.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
