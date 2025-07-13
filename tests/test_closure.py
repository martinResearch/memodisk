"""Testing memoization on closure."""

import tempfile
from typing import Callable

from memodisk import (
    get_last_cache_loading,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)


def function_closure(closure_x: int) -> Callable:
    @memoize
    def internal_function(y: int) -> int:
        return closure_x * y

    return internal_function


def function_closure2(closure_x: int) -> Callable:
    def internal_function(y: int) -> int:
        return closure_x * y

    return internal_function


@memoize
def function_creating_and_calling_closure(x: int, y: int) -> int:
    closure = function_closure2(x)
    return closure(y)  # type: ignore


def test_closure() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        closure = function_closure(2)

        assert closure(45) == 90

        # test caching works
        assert closure(45) == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".internal_function")

        # test cache invalidation works
        reset_last_cache_loading()
        closure2 = function_closure(3)
        assert closure2(45) == 135
        assert get_last_cache_loading() is None

        # test caching works calling again the first closure
        assert closure(45) == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".internal_function")


def test_closure2() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        assert function_creating_and_calling_closure(2, 45) == 90

        # test caching works
        assert function_creating_and_calling_closure(2, 45) == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".function_creating_and_calling_closure")


if __name__ == "__main__":
    test_closure()
    test_closure2()
