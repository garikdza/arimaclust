import functools
import logging
import time

import pandas as pd

logger = logging.getLogger('arimaclust')


def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug('%s took %.4f seconds', func.__name__, time.perf_counter() - start_time)
        return result

    return timeit_wrapper


def get_equal_time_periods(start_date, end_date, num_periods):
    """Split ``[start_date, end_date]`` into consecutive windows of ``num_periods`` months."""
    date_range = pd.date_range(start=start_date, end=end_date, freq=pd.DateOffset(months=num_periods))
    start_dates = date_range[:-1]
    end_dates = date_range[1:]
    return [(s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d')) for s, e in zip(start_dates, end_dates, strict=False)]
