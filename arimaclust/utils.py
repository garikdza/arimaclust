import functools
import logging
import time

logger = logging.getLogger('arimaclust')


def timeit(func):
    """Log at debug level how long each call of ``func`` takes."""
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug('%s took %.4f seconds', func.__name__, time.perf_counter() - start_time)
        return result

    return timeit_wrapper
