import numpy as np
import pytest

from arimaclust.data import generate_arma_sample


@pytest.fixture
def three_series() -> list[np.ndarray]:
    """Two AR(0.8) series and one AR(-0.5) series of different lengths: the first two are the close pair."""
    rng = np.random.default_rng(1)
    return [
        generate_arma_sample([0.8], [], 120, random_state=rng),
        generate_arma_sample([0.8], [], 100, random_state=rng),
        generate_arma_sample([-0.5], [], 90, random_state=rng),
    ]
