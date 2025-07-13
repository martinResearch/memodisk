"""Testing memoize on functions that also have other decorators than memoize."""

import functools
import tempfile
import types
from typing import Any, Callable, Optional, Union

import pytest

# from memodisk import get_alias_path
from non_pure_functions import call_func

from memodisk import get_last_cache_loading, memoize, set_cache_dir

global_a: int = 45


def function_using_global_variable(x: int) -> int:
    # alias = get_alias_path(inspect.currentframe())
    return x * global_a


def mydecorator(f_py: Optional[Callable]) -> Union[types.FunctionType, Callable]:
    assert callable(f_py) or f_py is None

    def _decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            # inspect.getclosurevars()
            return func(*args, **kwargs)

        return _wrapper

    return _decorator(f_py) if callable(f_py) else _decorator


@memoize
@mydecorator
def function_with_decorator() -> int:
    f = function_using_global_variable
    r = call_func(f, 2)
    return r  # type: ignore


@pytest.mark.skip(reason="decorator not supported yet")
def test_function_with_decorator() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert function_with_decorator() == 90

        # test caching works
        assert function_with_decorator() == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".function_with_decorator")


if __name__ == "__main__":
    test_function_with_decorator()
