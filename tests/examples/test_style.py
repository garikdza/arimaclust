import pytest

from style import BAND_COLOURS, CLUSTER_COLOURS, DISTANCE_RAMP, MINIMAL
from synthetic import BANDS


def _luminance(colour: str) -> float:
    red, green, blue = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def test_one_band_colour_per_band():
    assert len(BAND_COLOURS) == len(BANDS)


@pytest.mark.parametrize('ramp', [BAND_COLOURS, DISTANCE_RAMP])
def test_ordered_ramps_darken(ramp: tuple[str, ...]):
    luminances = [_luminance(colour) for colour in ramp]
    assert luminances == sorted(luminances, reverse=True)


def test_cluster_colours_are_distinct():
    assert len(set(CLUSTER_COLOURS)) == len(CLUSTER_COLOURS)


def test_minimal_holds_only_valid_matplotlib_settings():
    matplotlib = pytest.importorskip('matplotlib')
    # rc_context rejects an unknown key or a value of the wrong type
    with matplotlib.rc_context(MINIMAL):
        assert matplotlib.rcParams['axes.facecolor'] == MINIMAL['axes.facecolor']
