"""Debug test to understand tracer behavior under pytest."""
import tempfile
import time

from memodisk import get_last_cache_loading, memoize, reset_last_cache_loading, set_cache_dir


def helper():
    return time.time()


@memoize
def func_using_time():
    return helper()


def test_debug():
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp:
        set_cache_dir(tmp)
        reset_last_cache_loading()
        r1 = func_using_time()
        print(f"r1 = {r1}")
        reset_last_cache_loading()
        time.sleep(0.001)
        r2 = func_using_time()
        print(f"r2 = {r2}")
        print(f"cached: {get_last_cache_loading()}")
        assert get_last_cache_loading() is None
        assert r2 > r1
