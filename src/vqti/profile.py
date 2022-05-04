from timeit import default_timer
from typing import List, Dict, Any, Callable

def time_this(f):
    def timed_function(*args, **kwargs):
        ts = default_timer()
        print(f'Running {f.__name__} ...')

        result = f(*args, **kwargs)
        te = default_timer()
        t = 1000 * (te - ts)
        print(f'Completed {f.__name__} in {round(t, 3)} milliseconds')
        return result
    return timed_function


