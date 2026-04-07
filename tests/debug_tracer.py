"""Debug test to understand tracer behavior."""

import tempfile
import time

from memodisk import get_last_cache_loading, memoize, reset_last_cache_loading, set_cache_dir


def helper():
    return time.time()


@memoize
def func_using_time():
    return helper()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp:
        set_cache_dir(tmp)
        reset_last_cache_loading()
        r1 = func_using_time()
        print(f"r1 = {r1}")
        reset_last_cache_loading()
        r2 = func_using_time()
        print(f"r2 = {r2}")
        print(f"cached: {get_last_cache_loading()}")
