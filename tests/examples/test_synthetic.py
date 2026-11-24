import itertools
import math

import numpy as np
import pytest

from arimaclust.autoarima.estimation import to_arma_process
from arimaclust.metrics import piccolo
from synthetic import (
    CURATED_SETS,
    EXAMPLE1,
    EXAMPLE2,
    CuratedSet,
    band_of,
    clustering_scores,
    generate_separated_params,
    min_separation,
)

BAND_LABELS = {'δ < 0.2', '0.2 ≤ δ < 0.4', '0.4 ≤ δ < 0.6', 'δ ≥ 0.6'}
CURATED = {s.band: s.params for s in CURATED_SETS}


def test_min_separation_is_smallest_pair():
    # an AR(1) has pi = (phi, 0, ...), so the pairs are 1.0, 0.5 and 0.5 apart
    assert min_separation([([0.5], []), ([-0.5], []), ([], [])]) == pytest.approx(0.5)


@pytest.mark.parametrize(('delta', 'band'), [
    (0.0, 'δ < 0.2'),
    (0.1999, 'δ < 0.2'),
    (0.2, '0.2 ≤ δ < 0.4'),
    (0.4, '0.4 ≤ δ < 0.6'),
    (0.6, 'δ ≥ 0.6'),
    (5.0, 'δ ≥ 0.6'),
], ids=repr)
def test_band_of_boundaries(delta: float, band: str):
    assert band_of(delta) == band


@pytest.mark.parametrize('curated', CURATED_SETS, ids=lambda s: s.band)
def test_curated_set_processes_are_stationary_and_invertible(curated: CuratedSet):
    for params in curated.params:
        process = to_arma_process(*params)
        assert process.isstationary, params
        assert process.isinvertible, params


@pytest.mark.parametrize('curated', CURATED_SETS, ids=lambda s: s.band)
def test_curated_set_delta_is_recomputed_min_piccolo(curated: CuratedSet):
    recomputed = min(piccolo(to_arma_process(*a), to_arma_process(*b), tau=50)
                     for a, b in itertools.combinations(curated.params, 2))
    # the deltas are recorded to 4 dp
    assert curated.delta == pytest.approx(recomputed, abs=5e-5)
    low, high = {'δ < 0.2': (0.0, 0.2), '0.2 ≤ δ < 0.4': (0.2, 0.4), '0.4 ≤ δ < 0.6': (0.4, 0.6),
                 'δ ≥ 0.6': (0.6, math.inf)}[curated.band]
    assert low <= recomputed < high


def test_curated_sets_cover_every_band():
    assert {s.band for s in CURATED_SETS} == BAND_LABELS


@pytest.mark.parametrize('delta', [0.1, 0.5, 1.0])
def test_generated_sets_are_separated_stationary_invertible(delta: float):
    params = generate_separated_params(4, delta, seed=0)
    assert len(params) == 4
    processes = [to_arma_process(*p) for p in params]
    for p, process in zip(params, processes, strict=True):
        assert process.isstationary, p
        assert process.isinvertible, p
    for a, b in itertools.combinations(processes, 2):
        assert piccolo(a, b, tau=50) >= delta


def test_generator_is_reproducible_per_seed():
    params = generate_separated_params(3, 0.3, seed=7)
    assert params == generate_separated_params(3, 0.3, seed=7)
    assert params != generate_separated_params(3, 0.3, seed=8)


def test_generator_raises_when_search_limit_is_hit():
    # reciprocal roots are capped at modulus 0.8 with orders <= 2, so ||pi|| stays far below 500
    # and no pair can be 1000 apart
    with pytest.raises(RuntimeError, match='after 100 draws'):
        generate_separated_params(3, 1000.0, seed=0, max_draws=100)


def test_clustering_scores_recover_far_set():
    scores = clustering_scores(CURATED['δ ≥ 0.6'], length=400, n_per_cluster=5, seeds=(0, 1))
    assert scores == pytest.approx([1.0, 1.0])


def test_clustering_scores_reseed_each_draw():
    scores = clustering_scores(CURATED['δ < 0.2'], length=100, n_per_cluster=5, seeds=(0, 1, 2))
    np.testing.assert_array_equal(scores, clustering_scores(CURATED['δ < 0.2'], length=100, n_per_cluster=5,
                                                            seeds=(0, 1, 2)))
    # each draw gets its own seed, so the hard set scores differently on each
    assert len(set(scores)) > 1, scores


@pytest.mark.parametrize('params', EXAMPLE1 + EXAMPLE2)
def test_example_pairs_are_stationary_and_invertible(params: tuple[list[float], list[float]]):
    process = to_arma_process(*params)
    assert process.isstationary
    assert process.isinvertible


def test_example1_is_further_apart_than_example2():
    assert min_separation(EXAMPLE1) > min_separation(EXAMPLE2)
