"""Test memoization of functions using numba."""

import hashlib
import os
import sys

import numba
import numpy as np
from numba import njit

from memodisk import hashing_func_map, memoize, set_cache_dir

hashing_func_map[numba.core.registry.CPUDispatcher] = lambda x: hashlib.sha256(x.__code__.co_code).hexdigest()


@njit(cache=True)  # type: ignore
def square_array_numba(x: float, n: int) -> np.ndarray:
    return 2 * x * np.ones((n, n), dtype=np.float32)


@memoize
def square_array(x: float, n: int) -> np.ndarray:
    r = square_array_numba(x, n)
    return r  # type: ignore


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tmp_folder = sys.argv[1]
    else:
        tmp_folder = os.path.dirname(__file__)
    set_cache_dir(tmp_folder)
    r = square_array(5, 3)
    assert np.allclose(r, np.full((3, 3), 10.0))
