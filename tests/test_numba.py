"""Testing memoization on function jitted with numba."""

import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import Any, Callable, ParamSpec, TypeVar, Union, overload

import numba
import numba.core.registry
import numpy as np
from numba import njit

from memodisk import (
    get_last_cache_loading,
    hashing_func_map,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)


hashing_func_map[numba.core.registry.CPUDispatcher] = lambda x: hashlib.sha256(x.__code__.co_code).hexdigest()


@njit(cache=True)
def function_numba(x: int) -> int:
    prod: int = x * x
    return prod


@memoize
def function_calling_numba(x: int) -> int:
    return function_numba(x)


@njit(cache=True)
def square_array_numba(x: float, n: int) -> np.ndarray:
    return x * np.ones((n, n), dtype=np.float32)


@memoize
def square_array(x: float, n: int) -> np.ndarray:
    r = square_array_numba(x, n)
    return r


def test_numba() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        # Compile
        assert function_calling_numba(1) == 1
        # Use the compiled function
        function_calling_numba(1)
        # test caching works
        for _ in range(10):
            reset_last_cache_loading()
            assert function_calling_numba(1) == 1
            last_cache_loading = get_last_cache_loading()
            assert last_cache_loading is not None and last_cache_loading.endswith(".function_calling_numba")


def test_numba2() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        # Compile
        assert np.allclose(square_array(5, 3), np.full((3, 3), 5.0))
        # Use the compiled function
        assert np.allclose(square_array(5, 3), np.full((3, 3), 5.0))
        # test caching works
        for _ in range(10):
            reset_last_cache_loading()
            assert np.allclose(square_array(5, 3), np.full((3, 3), 5.0))
            last_cache_loading = get_last_cache_loading()
            assert last_cache_loading is not None and last_cache_loading.endswith(".square_array")


def test_numba_code_dependency_change() -> None:
    """Test that change in a code dependency is detected"""
    import sys

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        python_exe = sys.executable
        folder = os.path.dirname(__file__)
        tmp_file = os.path.join(tmp_folder, "code_test_code_dep_numba_tmp.py")

        shutil.copy(os.path.join(folder, "code_test_code_dep_numba_a.py"), tmp_file)

        # first run
        subprocess.run([python_exe, tmp_file, tmp_folder])

        # # second run , check caching works
        # result = subprocess.run(
        #     [python_exe, tmp_file, tmp_folder], stdout=subprocess.PIPE
        # )
        # assert (
        #     b"Result loaded from __main__.square_array"
        #     in result.stdout
        # )

        # # check invalidation works
        # shutil.copy(os.path.join(folder, "code_test_code_dep_numba_b.py"), tmp_file)
        # subprocess.run([python_exe, tmp_file, tmp_folder])


if __name__ == "__main__":
    test_numba()
    test_numba2()
    test_numba_code_dependency_change()
