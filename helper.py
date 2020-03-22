import time
from datetime import timedelta


def time_measurement(func):

    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        finish_time = time.perf_counter()
        elapsed_time = finish_time - start_time

        print(f'{func.__name__}() elapsed time: {timedelta(seconds=elapsed_time)}\n')

        return result

    return wrapper
