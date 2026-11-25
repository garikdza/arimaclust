"""
Clusters of the FRED-MD macroeconomic series: the 121 series with no gap in 1960-2019, compared by the cepstral
distance of their fitted ARMA models and clustered with ``KMedoids(n_clusters=4)``.

It prints each cluster's size, medoid, transformation codes, FRED-MD groups, median sample autocorrelation and
members, and draws the autocorrelation profile of each cluster (``figures/autocorrelation_profiles.png``) and the
distance matrix ordered by cluster beside its 2-D MDS embedding (``figures/distance_matrix.png``).

The first run downloads the vintage into ``data/``, so it needs the network, and fits for about 90 seconds. The
results are saved to ``figures/clusters.npz`` and later runs plot from them; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/fred_md/clusters.py``.
"""

import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from statsmodels.tsa.stattools import acf

from arimaclust import ARMADistance, KMedoids

# results.py and style.py are shared by every example and sit one folder up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fred_md import GROUP_NAMES, fetch_vintage, group_ids, prepare, read_vintage
from results import load_or_compute, parse_args
from style import AXIS, CLUSTER_COLOURS, DISTANCE_RAMP, INK, INK_SOFT, MINIMAL, SURFACE

N_CLUSTERS = 4
# the autocorrelation profiles run over the first year of lags
LAGS = 12
HERE = Path(__file__).parent
VINTAGE = HERE / 'data' / '2026-01-md.csv'
FIGURES = HERE / 'figures'
RESULTS = FIGURES / 'clusters.npz'


def compute() -> dict[str, np.ndarray]:
    """Download the vintage unless cached, cluster its prepared series and take their sample autocorrelations."""
    fetch_vintage(VINTAGE)
    data, tcodes = read_vintage(VINTAGE)
    series, dropped = prepare(data, tcodes)
    print(f'fitting {series.shape[1]} series of {len(series)} months ...')
    distance = ARMADistance(max_p=3, max_q=3, d=0, metric='cepstral', tau=50, n_jobs=4)
    model = KMedoids(n_clusters=N_CLUSTERS, distance=distance).fit(series)
    autocorrelations = [acf(series[name].to_numpy(), nlags=LAGS, fft=True)[1:] for name in series.columns]
    return {
        'names': np.array(series.columns, dtype=str),
        'tcodes': tcodes[series.columns].to_numpy(),
        'groups': group_ids(series.columns),
        'distances': model.distance_.dist_matrix_.to_numpy(),
        'labels': model.labels_,
        'medoids': model.medoids_,
        'acf': np.array(autocorrelations),
        'dropped': np.array(dropped, dtype=str),
        'vintage': np.array(VINTAGE.name),
        'start': np.array(series.index[0].strftime('%Y-%m')),
        'end': np.array(series.index[-1].strftime('%Y-%m')),
        'length': np.array(len(series)),
    }


def plot(results: dict[str, np.ndarray]) -> None:
    """Print the saved clusters and draw their autocorrelation profiles and their distance matrix."""
    ami = adjusted_mutual_info_score(results['groups'], results['labels'])
    _print_clusters(results, ami)
    _draw_profiles(results)
    _draw_distances(results, ami)


def _counts(values: np.ndarray, label: Callable[[int], str]) -> str:
    """Count the distinct ``values``, most common first, as ``'<label> <count>'`` pairs."""
    return ', '.join(f'{label(value)} {count}' for value, count in Counter(values.tolist()).most_common())


def _print_clusters(results: dict[str, np.ndarray], ami: float) -> None:
    """Print the data, the scores and each cluster's composition, autocorrelations and members."""
    names, labels, acfs = results['names'], results['labels'], results['acf']
    print(
        f'FRED-MD {results["vintage"]}, {results["start"]} to {results["end"]}: '
        f'T={int(results["length"])}, N={len(names)}; dropped for a gap in the window: {", ".join(results["dropped"])}'
    )
    silhouette = silhouette_score(results['distances'], labels, metric='precomputed')
    print(f'AMI against the {len(GROUP_NAMES)} FRED-MD groups {ami:.3f}, silhouette {silhouette:.3f}')
    for k, medoid in enumerate(results['medoids']):
        members = labels == k
        print(f'\ncluster {k + 1}: {members.sum()} series, medoid {names[medoid]}')
        print(f'  tcodes   {_counts(results["tcodes"][members], lambda tcode: f"tc{tcode}")}')
        print(f'  groups   {_counts(results["groups"][members], GROUP_NAMES.__getitem__)}')
        for lag in (1, 2, LAGS):
            q25, median, q75 = np.quantile(acfs[members, lag - 1], [0.25, 0.5, 0.75])
            print(f'  ACF lag {lag:<2} median {median:+.2f}  IQR [{q25:+.2f}, {q75:+.2f}]')
        print(f'  members  {", ".join(names[members])}')


def _draw_profiles(results: dict[str, np.ndarray]) -> None:
    """Draw one panel per cluster: the median sample autocorrelation of its members by lag, with its IQR."""
    names, labels, medoids, acfs = results['names'], results['labels'], results['medoids'], results['acf']
    lags = np.arange(1, acfs.shape[1] + 1)
    with plt.rc_context(MINIMAL):
        fig, axes = plt.subplots(1, len(medoids), figsize=(12, 3.6), sharey=True, layout='constrained')
        for k, (ax, medoid, colour) in enumerate(zip(axes, medoids, CLUSTER_COLOURS[: len(medoids)], strict=True)):
            members = acfs[labels == k]
            q25, median, q75 = np.quantile(members, [0.25, 0.5, 0.75], axis=0)
            ax.axhline(0, color=AXIS, linewidth=0.8)
            ax.fill_between(lags, q25, q75, color=colour, alpha=0.15, linewidth=0)
            ax.plot(lags, median, color=colour)
            ax.scatter(lags, median, s=30, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.set_xticks([1, 3, 6, 9, 12])
            ax.set_xlabel('lag (months)')
            ax.set_title(f'cluster {k + 1}: {len(members)} series\nmedoid {names[medoid]}')
        axes[0].set_ylabel('sample autocorrelation')
        fig.suptitle('sample autocorrelation of the members of each cluster: median and interquartile range')
        _save(fig, 'autocorrelation_profiles.png')


def _draw_distances(results: dict[str, np.ndarray], ami: float) -> None:
    """Draw the heatmap of the distance matrix ordered by cluster beside its MDS embedding coloured by cluster."""
    D, names, labels, medoids = results['distances'], results['names'], results['labels'], results['medoids']
    order = np.argsort(labels, kind='stable')
    sizes = np.bincount(labels)
    ends = np.cumsum(sizes)
    cluster_names = [f'cluster {k + 1}' for k in range(len(sizes))]

    with plt.rc_context(MINIMAL):
        fig, (heat, mds) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={'width_ratios': [1.1, 1]})
        heat.grid(False)
        image = heat.imshow(
            D[np.ix_(order, order)],
            cmap=LinearSegmentedColormap.from_list('distance', DISTANCE_RAMP),
            interpolation='nearest',
        )
        for boundary in ends[:-1] - 0.5:
            heat.axhline(boundary, color=SURFACE, linewidth=2)
            heat.axvline(boundary, color=SURFACE, linewidth=2)
        centres = ends - sizes / 2 - 0.5
        heat.set_xticks(centres, cluster_names)
        heat.set_yticks(centres, cluster_names)
        heat.spines[['left', 'bottom']].set_visible(False)
        bar = fig.colorbar(image, ax=heat, fraction=0.046, pad=0.03)
        bar.outline.set_visible(False)
        bar.set_label('cepstral distance (lighter = closer)')

        embedding = MDS(n_components=2, dissimilarity='precomputed', n_init=4, random_state=0).fit_transform(D)
        mds.grid(False)
        for k, colour in zip(range(len(sizes)), CLUSTER_COLOURS[: len(sizes)], strict=True):
            mds.scatter(
                *embedding[labels == k].T,
                s=40,
                color=colour,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=f'{cluster_names[k]} ({sizes[k]})',
                zorder=3,
            )
        mds.scatter(
            *embedding[medoids].T, s=130, facecolor='none', edgecolor=INK, linewidth=1, zorder=4, label='medoid'
        )
        for medoid in medoids:
            mds.annotate(
                names[medoid], embedding[medoid], xytext=(6, 6), textcoords='offset points', fontsize=8, color=INK_SOFT
            )
        mds.set_xticks([])
        mds.set_yticks([])
        mds.spines[['left', 'bottom']].set_visible(False)
        mds.legend(
            loc='upper left', bbox_to_anchor=(1.01, 1), title=f'AMI {ami:.2f} against the {len(GROUP_NAMES)} groups'
        )
        _save(fig, 'distance_matrix.png')


def _save(fig: plt.Figure, name: str) -> None:
    """Write ``fig`` to ``FIGURES / name`` and close it."""
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
