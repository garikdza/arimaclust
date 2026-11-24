from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from results import load_or_compute, load_results, parse_args, save_results


def test_round_trip_keeps_values_dtypes_and_shapes(tmp_path: Path):
    saved = {
        'bands': np.array(['δ < 0.2', '0.2 ≤ δ < 0.4']),
        'delta': np.array(0.3055),
        'n_per_cluster': np.array(10),
        'scores': np.random.default_rng(0).random((2, 3, 4)),
        'y_true': np.array([0, 0, 1]),
    }
    path = tmp_path / 'r.npz'
    save_results(path, saved)
    loaded = load_results(path)
    assert loaded.keys() == saved.keys()
    for key, value in saved.items():
        assert_array_equal(loaded[key], value)
        assert loaded[key].dtype == value.dtype, key
        assert loaded[key].shape == value.shape, key


def test_save_writes_exactly_the_given_path_and_creates_its_folder(tmp_path: Path):
    path = tmp_path / 'figures' / 'r.npz'
    save_results(path, {'x': np.array([1.0])})
    assert path.exists()


def test_save_rejects_object_arrays(tmp_path: Path):
    path = tmp_path / 'r.npz'
    with pytest.raises(ValueError):
        save_results(path, {'pairs': np.array([[1], [1, 2]], dtype=object)})
    assert list(tmp_path.iterdir()) == []


def test_failed_save_keeps_the_previous_file(tmp_path: Path):
    path = tmp_path / 'r.npz'
    save_results(path, {'x': np.array([1.0])})
    with pytest.raises(ValueError):
        save_results(path, {'pairs': np.array([[1], [1, 2]], dtype=object)})
    assert list(tmp_path.iterdir()) == [path]
    assert_array_equal(load_results(path)['x'], [1.0])


def test_load_or_compute_computes_and_saves_when_file_missing(tmp_path: Path):
    path = tmp_path / 'r.npz'
    calls = []

    def compute() -> dict[str, np.ndarray]:
        calls.append(1)
        return {'x': np.array([1.0, 2.0])}

    results = load_or_compute(path, compute, recompute=False)
    assert len(calls) == 1
    assert path.exists()
    assert results.keys() == {'x'}
    assert_array_equal(results['x'], [1.0, 2.0])


def test_load_or_compute_loads_without_computing_when_file_present(tmp_path: Path):
    path = tmp_path / 'r.npz'
    save_results(path, {'x': np.array([1.0])})

    def compute() -> dict[str, np.ndarray]:
        raise AssertionError('must not fit')

    assert_array_equal(load_or_compute(path, compute, recompute=False)['x'], [1.0])


def test_load_or_compute_recomputes_and_overwrites_when_asked(tmp_path: Path):
    path = tmp_path / 'r.npz'
    save_results(path, {'x': np.array([1.0])})
    results = load_or_compute(path, lambda: {'x': np.array([2.0])}, recompute=True)
    assert_array_equal(results['x'], [2.0])
    assert_array_equal(load_results(path)['x'], [2.0])


@pytest.mark.parametrize(('argv', 'recompute'), [([], False), (['--recompute'], True)], ids=repr)
def test_parse_args_reads_recompute_flag(argv: list[str], recompute: bool):
    assert parse_args('an example script', argv).recompute is recompute
