"""Testing class method memoization."""

import tempfile
from typing import Any

import non_pure_functions

from memodisk import (
    get_last_cache_loading,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)


class ClassExample:
    """Simple class."""

    def __init__(self, v: int):
        self.v = v

    def __call__(self, x: int) -> int:
        return self.v * x


@memoize
def f(x: int, y: int) -> int:
    c = ClassExample(x)
    return c(y)


class ClassExample2:
    """Simple class used in the tests."""

    def __init__(self, v: int):
        self.v = v

    def set_v(self, v: int) -> None:
        self.v = v

    @memoize
    def __call__(self, x: int) -> int:
        return self.v * x


@memoize
def call_method(cl: Any, x: Any) -> Any:
    return cl(x)


def test_class_dependency() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        assert f(5, 6) == 30

        # test caching works
        assert f(5, 6) == 30
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".f")


def test_class_method() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        c = ClassExample2(2)
        assert c(6) == 12

        # test caching works
        assert c(6) == 12
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith("ClassExample2.__call__")

        # test cache invalidation works
        reset_last_cache_loading()
        c.set_v(5)
        assert c(6) == 30
        assert get_last_cache_loading() is None


def test_class_method_using_globals() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        c = non_pure_functions.ClassExample4()
        assert call_method(c, 2) == 90

        # test caching works
        assert call_method(c, 2) == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith("call_method")

        # test cache invalidation works
        reset_last_cache_loading()

        non_pure_functions.global_list = [40]

        assert call_method(c, 2) == 80
        assert get_last_cache_loading() is None


if __name__ == "__main__":
    test_class_method_using_globals()
    test_class_method()
    test_class_dependency()
