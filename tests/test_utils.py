import logging
import re

import numpy as np
import pytest

from arimaclust.utils import check_tau, timeit


@pytest.mark.parametrize('tau', [1, 50, np.int64(3)], ids=repr)
def test_check_tau_accepts_positive_integers(tau: object):
    check_tau(tau)


@pytest.mark.parametrize('tau', [0, -1, 2.5, True, '3', None], ids=repr)
def test_check_tau_rejects_the_rest(tau: object):
    with pytest.raises(ValueError, match=rf'tau must be an integer >= 1, got {re.escape(repr(tau))}'):
        check_tau(tau)


def test_timeit_returns_the_result_and_logs_the_call(caplog: pytest.LogCaptureFixture):
    @timeit
    def add(a: int, b: int = 0) -> int:
        return a + b

    with caplog.at_level(logging.DEBUG, logger='arimaclust'):
        assert add(1, b=2) == 3
    assert add.__name__ == 'add'
    assert re.fullmatch(r'add took \d+\.\d{4} seconds', caplog.records[-1].getMessage())
