"""
Clustering accuracy against separation δ on generated two-cluster sets (``figures/accuracy_vs_delta.png``).

Each band gets sets from ``generate_separated_params`` whose δ falls inside it. The spread within a band is wide,
because δ is not the only thing that makes a pair of processes hard to tell apart from short series.

Run with ``uv run --extra examples python examples/accuracy_vs_delta.py``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthetic import BANDS, TAU, band_of, clustering_scores, generate_separated_params, min_separation

LENGTH = 200
N_PER_CLUSTER = 10
SEEDS = range(4)
SETS_PER_BAND = 6
# with two clusters δ is the only pairwise distance; a far third cluster would lift every band's accuracy
N_CLUSTERS = 2
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, (low, _)) in enumerate(BANDS.items()):
        deltas, scores = [], []
        seed = 1000 * i
        while len(deltas) < SETS_PER_BAND:
            params = generate_separated_params(N_CLUSTERS, low, seed)
            delta = min_separation(params)
            if band_of(delta) == label:
                score = clustering_scores(params, LENGTH, N_PER_CLUSTER, SEEDS).mean()
                deltas.append(delta)
                scores.append(score)
                print(f'{label:>14}  seed={seed:<5} δ={delta:.4f}  AMI {score:.3f}')
            seed += 1
        mean = np.mean(scores)
        print(f'{label:>14}  mean AMI {mean:.3f}')
        points = ax.scatter(deltas, scores, label=label)
        ax.hlines(mean, min(deltas), max(deltas), color=points.get_facecolor(), linewidth=2)
    for boundary, _ in list(BANDS.values())[1:]:
        ax.axvline(boundary, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(f'separation δ (smallest Piccolo distance, tau={TAU})')
    ax.set_ylabel(f'AMI (mean over {len(SEEDS)} draws)')
    ax.set_title(f'generate_separated_params sets, {N_CLUSTERS} clusters × {N_PER_CLUSTER} series at T={LENGTH}\n'
                 'dots: one set each; bars: band mean over its sets')
    ax.legend(title='separation band')

    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / 'accuracy_vs_delta.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
