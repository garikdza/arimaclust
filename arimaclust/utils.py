import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import functools
import time


def timeit(func):
    @functools.wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def get_equal_time_periods(start_date, end_date, num_periods):
    date_range = pd.date_range(start=start_date, end=end_date, freq=pd.DateOffset(months=num_periods))
    start_dates = date_range[:-1]
    end_dates = date_range[1:]
    return [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) for start_date, end_date in
            zip(start_dates, end_dates)]
