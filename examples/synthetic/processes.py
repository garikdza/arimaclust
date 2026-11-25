"""
The curated sets themselves, no fitting: one sample series per cluster (``figures/sample_series.png``) and the true
AR(∞) coefficients π₁…π_tau that the Piccolo distance compares (``figures/ar_coefficients.png``).

Run with ``uv run --extra examples python examples/synthetic/processes.py``.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arimaclust.data import generate_arma_sample
from arimaclust.modeling import ARIMA

# results.py and style.py are shared by every example and sit one folder up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import AXIS, CLUSTER_COLOURS, MINIMAL
from synthetic import CURATED_SETS, TAU

LENGTH = 200
# vertical gap between the series drawn in one panel
OFFSET = 6
LAG_TICKS = (1, 2, 5, 10, 20, 50)
FIGURES = Path(__file__).parent / 'figures'


def _save(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to ``FIGURES / name`` and close it."""
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def _sample_series() -> None:
    """Draw one sample series per cluster of each curated set, offset vertically."""
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, curated in zip(axes.flat, CURATED_SETS, strict=True):
        for k, ((ar, ma), colour) in enumerate(zip(curated.params, CLUSTER_COLOURS, strict=True)):
            ax.plot(
                generate_arma_sample(ar, ma, LENGTH, random_state=rng) - k * OFFSET,
                color=colour,
                linewidth=0.8,
                label=f'AR {ar}, MA {ma}',
            )
        ax.set_yticks([])
        # headroom for the legend above the top series
        ax.set_ylim(top=ax.get_ylim()[1] + OFFSET)
        ax.set_title(f'{curated.band} (δ={curated.delta})')
        ax.legend(fontsize=8, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('t')
    fig.suptitle(f'one sample series of length {LENGTH} per cluster, offset vertically')
    _save(fig, 'sample_series.png')


def _ar_coefficients() -> None:
    """Draw the true AR(∞) coefficients of each cluster of each curated set."""
    lags = np.arange(1, TAU + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, curated in zip(axes.flat, CURATED_SETS, strict=True):
        for (ar, ma), colour in zip(curated.params, CLUSTER_COLOURS, strict=True):
            ax.plot(lags, ARIMA(ar, ma).pi(TAU), color=colour, marker='.', linewidth=1, label=f'AR {ar}, MA {ma}')
        ax.axhline(0, color=AXIS, linewidth=0.8)
        # log lags, so the first few, where these low-order processes differ, are not squeezed together
        ax.set_xscale('log')
        ax.set_xticks(LAG_TICKS, [str(k) for k in LAG_TICKS])
        ax.minorticks_off()
        ax.set_title(f'{curated.band} (δ={curated.delta})')
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('lag k')
    for ax in axes[:, 0]:
        ax.set_ylabel('π_k')
    fig.suptitle(
        f'true AR(∞) coefficients π₁…π_{TAU} of each cluster; δ is the smallest Euclidean distance between them'
    )
    _save(fig, 'ar_coefficients.png')


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    with plt.rc_context(MINIMAL):
        _sample_series()
        _ar_coefficients()


if __name__ == '__main__':
    main()
