"""File used to test change in code dependency."""

import sys
import time

from memodisk import memoize, set_cache_dir


def fun_a(x: int) -> int:
    print("executing fun_a")
    return x * 3


@memoize
def fun_b(x: int) -> int:
    print("sleep 1")
    time.sleep(1)
    print("executing fun_b")
    return fun_a(x) + 2


if __name__ == "__main__":
    tmp_folder = sys.argv[1]
    set_cache_dir(tmp_folder)
    assert fun_b(5) == 17
