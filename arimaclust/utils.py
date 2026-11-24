import functools
import logging
import numbers
import time

logger = logging.getLogger('arimaclust')


def check_tau(tau: object) -> None:
    """Raise a ``ValueError`` unless ``tau``, the number of AR(inf) coefficients compared, is an integer >= 1."""
    if not isinstance(tau, numbers.Integral) or isinstance(tau, bool) or tau < 1:
        raise ValueError(f'tau must be an integer >= 1, got {tau!r}')


def timeit(func):
    """Log at debug level how long each call of ``func`` takes."""

    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug('%s took %.4f seconds', func.__name__, time.perf_counter() - start_time)
        return result

    return timeit_wrapper
