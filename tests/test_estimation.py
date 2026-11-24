from types import SimpleNamespace

import numpy as np
import pmdarima
import pytest

from arimaclust.autoarima.estimation import estimate_processes, to_arma_process
from arimaclust.data import generate_arma_sample


def test_to_arma_process_negates_ar_params_only():
    process = to_arma_process([0.5, -0.2], [0.3])
    # AR params enter the lag polynomial negated, 1 - 0.5B + 0.2B^2; MA params keep their sign, 1 + 0.3B
    np.testing.assert_allclose(process.ar, [1, -0.5, 0.2])
    np.testing.assert_allclose(process.ma, [1, 0.3])
    np.testing.assert_allclose(process.arcoefs, [0.5, -0.2])
    np.testing.assert_allclose(process.macoefs, [0.3])


def test_to_arma_process_keeps_differencing_in_ar_polynomial():
    # (1 - 0.5B)(1 - B)^2 = 1 - 2.5B + 2B^2 - 0.5B^3
    process = to_arma_process([0.5], [0.3], d=2)
    np.testing.assert_allclose(process.ar, [1, -2.5, 2, -0.5])
    np.testing.assert_allclose(process.ma, [1, 0.3])
    assert not process.isstationary


def test_estimate_processes_returns_arma_part_and_d():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal(400))
    noise = generate_arma_sample([], [], 400, random_state=rng)
    (walk_arma, walk_d), (noise_arma, noise_d) = estimate_processes([walk, noise], d=None, max_p=2, max_q=2)
    assert (walk_d, noise_d) == (1, 0)
    assert walk_arma.isstationary


def test_failed_fit_names_series(monkeypatch: pytest.MonkeyPatch):
    def fail_on_short(x: np.ndarray, **kwargs) -> SimpleNamespace:
        if len(x) == 20:
            raise ValueError('Could not successfully fit a viable ARIMA model')
        return SimpleNamespace(arparams=lambda: np.array([]), maparams=lambda: np.array([]), order=(0, 0, 0))

    monkeypatch.setattr(pmdarima, 'auto_arima', fail_on_short)
    with pytest.raises(ValueError, match=r"could not fit series 'b': Could not successfully fit"):
        estimate_processes([np.zeros(30), np.zeros(20)], names=['a', 'b'])
