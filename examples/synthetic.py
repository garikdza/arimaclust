"""
Synthetic ARMA cluster sets at a controlled separation δ, shared by the example scripts.

δ is the smallest Piccolo distance (``tau=50``) between the processes that generate the clusters.
"""
import itertools
import math
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from arimaclust import ARMADistance, ConsistentClustering
from arimaclust.data import generate_clusters
from arimaclust.metrics import piccolo
from arimaclust.modeling import FittedARIMA

# (ar_params, ma_params), without the zero lag
Params = tuple[list[float], list[float]]

TAU = 50
# generated processes have AR and MA orders up to MAX_ORDER, with reciprocal roots of modulus up to MAX_ROOT_MODULUS
MAX_ORDER = 2
MAX_ROOT_MODULUS = 0.8

BANDS: dict[str, tuple[float, float]] = {
    'δ < 0.2': (0.0, 0.2),
    '0.2 ≤ δ < 0.4': (0.2, 0.4),
    '0.4 ≤ δ < 0.6': (0.4, 0.6),
    'δ ≥ 0.6': (0.6, math.inf),
}


def band_of(delta: float) -> str:
    """Return the label of the band in ``BANDS`` that holds ``delta``."""
    return next(label for label, (low, high) in BANDS.items() if low <= delta < high)


class CuratedSet(NamedTuple):
    """Cluster-generating parameters whose smallest pairwise Piccolo distance (``tau=50``) is ``delta``."""
    band: str
    delta: float
    params: list[Params]


def min_separation(params: list[Params]) -> float:
    """
    Smallest pairwise Piccolo distance between the processes given by ``params``.

    Parameters
    ----------
    params : list[Params]
        ``(ar_params, ma_params)`` of each process, at least two

    Returns
    -------
    delta : float
        the separation δ of the set
    """
    models = [FittedARIMA(*p) for p in params]
    return min(piccolo(a, b, tau=TAU) for a, b in itertools.combinations(models, 2))


# one set per band, each mixing AR, MA and ARMA clusters; the deltas are recorded to 4 dp and checked in the tests
CURATED_SETS: tuple[CuratedSet, ...] = (
    CuratedSet('δ < 0.2', 0.1008, [([0.5], [-0.3]), ([0.3], []), ([], [0.2])]),
    CuratedSet('0.2 ≤ δ < 0.4', 0.3055, [([0.6], []), ([0.1], [0.2]), ([], [0.5])]),
    CuratedSet('0.4 ≤ δ < 0.6', 0.5016, [([0.5], [0.1]), ([0.4, 0.4], []), ([], [0.1])]),
    CuratedSet('δ ≥ 0.6', 0.8010, [([-0.6], []), ([-0.2, 0.7], []), ([], [0.2])]),
)


# two processes whose AR(∞) representations are far apart
EXAMPLE1: list[Params] = [([0.3, -0.2], [-0.5]), ([0.7, -0.4, 0.1], [0.5, -0.2])]
# two processes whose AR(∞) representations are near: same MA part, AR parts 0.05 apart
EXAMPLE2: list[Params] = [([0.3, -0.2], [-0.5]), ([0.25, -0.15], [-0.5])]


def _lag_coefficients(rng: np.random.Generator, order: int) -> np.ndarray:
    """Coefficients after the leading 1 of a random (1 - r_1 B)...(1 - r_order B) with |r_i| <= MAX_ROOT_MODULUS."""
    if order == 0:
        return np.empty(0)
    if order == 2 and rng.random() < 0.5:
        modulus, angle = rng.uniform(0.2, MAX_ROOT_MODULUS), rng.uniform(0, np.pi)
        roots = modulus * np.exp([1j * angle, -1j * angle])
    else:
        roots = rng.uniform(-MAX_ROOT_MODULUS, MAX_ROOT_MODULUS, size=order)
    return np.poly(roots).real[1:]


def _draw_params(rng: np.random.Generator) -> Params:
    """Draw a random stationary, invertible ARMA process that is not white noise."""
    p, q = 0, 0
    while p == q == 0:
        p, q = rng.integers(0, MAX_ORDER, endpoint=True, size=2)
    # the AR polynomial is 1 - sum phi_k B^k and the MA polynomial 1 + sum theta_k B^k;
    # rounding to 3 dp moves no root of modulus <= 0.8 out to the unit circle
    ar = np.round(-_lag_coefficients(rng, p), 3).tolist()
    ma = np.round(_lag_coefficients(rng, q), 3).tolist()
    return ar, ma


def generate_separated_params(n_clusters: int, delta: float, seed: int, max_draws: int = 10_000) -> list[Params]:
    """
    Generate cluster-generating ARMA parameters with pairwise Piccolo distance (``tau=50``) of at least ``delta``.

    Every process is stationary and invertible, with AR and MA orders up to ``MAX_ORDER``. The same seed gives
    the same result. The search is greedy: it keeps each random draw that is at least ``delta`` from the ones
    kept so far, so it can give up on requests that are feasible but hard.

    Parameters
    ----------
    n_clusters : int
        number of processes
    delta : float
        smallest allowed pairwise Piccolo distance
    seed : int
        seed of the random draws
    max_draws : int
        number of draws after which the search gives up

    Returns
    -------
    params : list[Params]
        ``(ar_params, ma_params)`` of each process

    Raises
    ------
    RuntimeError
        if ``max_draws`` draws don't give ``n_clusters`` separated processes
    """
    rng = np.random.default_rng(seed)
    kept, models = [], []
    for _ in range(max_draws):
        candidate = _draw_params(rng)
        model = FittedARIMA(*candidate)
        if all(piccolo(model, other, tau=TAU) >= delta for other in models):
            kept.append(candidate)
            models.append(model)
            if len(kept) == n_clusters:
                return kept
    raise RuntimeError(f'found {len(kept)} of {n_clusters} processes with pairwise Piccolo distance >= {delta} '
                       f'after {max_draws} draws; lower delta or n_clusters, or raise max_draws')


def clustering_scores(params: list[Params], length: int, n_per_cluster: int, seeds: Iterable[int]) -> np.ndarray:
    """
    Cluster freshly drawn series once per seed and score each clustering against the true labels.

    Each draw has ``n_per_cluster`` series of ``length`` per process, clustered by ``ConsistentClustering``
    with the default ``ARMADistance`` (Piccolo, BIC).

    Parameters
    ----------
    params : list[Params]
        ``(ar_params, ma_params)`` of each cluster-generating process
    length : int
        length of every series
    n_per_cluster : int
        number of series per cluster
    seeds : Iterable[int]
        one seed per draw

    Returns
    -------
    scores : np.ndarray
        adjusted mutual information of each draw
    """
    scores = []
    for seed in seeds:
        X, y = generate_clusters(params, length, n_per_cluster, random_state=seed)
        labels = ConsistentClustering(n_clusters=len(params), distance=ARMADistance(n_jobs=-1)).fit_predict(X)
        scores.append(adjusted_mutual_info_score(y, labels))
    return np.array(scores)
