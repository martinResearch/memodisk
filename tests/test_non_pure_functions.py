"""Testing memoization on non-pure funtions."""

import tempfile
from typing import Any, Callable, Dict

import non_pure_functions
from non_pure_functions import call_func
from non_pure_functions import (
    function_using_global_variable_5 as function_using_global_variable_ext,
)

from memodisk import (
    get_last_cache_loading,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)

# import inspect

global_a: int = 45


def function_using_global_variable(x: int) -> int:
    return x * global_a


@memoize
def function_with_dependency_using_global_variable1(x: int) -> int:
    return function_using_global_variable(x)


@memoize
def function_with_dependency_using_global_variable2(x: int) -> int:
    return function_using_global_variable_ext(x)  # type: ignore


@memoize
def function_using_build_in_functions() -> int:
    from math import ceil

    return ceil(8.2)


def test_global_variable_dependency_ext() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        non_pure_functions.global_list[0] = 45

        assert function_with_dependency_using_global_variable2(1) == 45

        # test caching works
        assert function_with_dependency_using_global_variable2(1) == 45
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None and last_cache_loading.endswith(
            ".function_with_dependency_using_global_variable2"
        )

        # test cache invalidation works
        reset_last_cache_loading()
        non_pure_functions.global_list[0] = 40
        assert function_with_dependency_using_global_variable2(1) == 40
        assert get_last_cache_loading() is None


def test_global_variable_dependency() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        assert function_with_dependency_using_global_variable1(1) == 45

        # test caching works
        assert function_with_dependency_using_global_variable1(1) == 45
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None and last_cache_loading.endswith(
            ".function_with_dependency_using_global_variable1"
        )

        # test cache invalidation works
        reset_last_cache_loading()
        global global_a
        value_before = global_a
        global_a = 40
        assert function_with_dependency_using_global_variable1(1) == 40
        assert get_last_cache_loading() is None
        global_a = value_before  # putting the global  variable back to its original value for the other tests


def test_pure_function_buildin_functions() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        function_using_build_in_functions()


def fun_a() -> str:
    return "a"


def fun_b() -> str:
    return "b"


glob_fun = fun_a


@memoize
def function_using_global_func_variable() -> str:
    return glob_fun()


@memoize
def function_using_alias() -> int:
    f = function_using_global_variable
    r = f(2)
    return r


@memoize
def function_using_injection() -> int:
    f = function_using_global_variable
    r = call_func(f, 2)
    return r  # type: ignore


def test_function_using_global_func_variable() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert function_using_global_func_variable() == "a"

        # test caching works
        assert function_using_global_func_variable() == "a"
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".function_using_global_func_variable")

        # test cache invalidation works
        reset_last_cache_loading()
        global glob_fun
        glob_fun = fun_b
        assert function_using_global_func_variable() == "b"
        assert get_last_cache_loading() is None


def test_function_using_alias() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert function_using_alias() == 90

        # test caching works
        assert function_using_alias() == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".function_using_alias")


def test_function_using_injection() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert function_using_injection() == 90

        # test caching works
        assert function_using_injection() == 90
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".function_using_injection")


def cached_one_arg_func(user_function: Callable) -> Callable:
    sentinel = object()  # unique object used to signal cache misses
    cache: Dict[Any, Any] = {}
    cache_get = cache.get

    def wrapper(key: Any) -> Any:
        result = cache_get(key, sentinel)
        if result is not sentinel:
            return result
        result = user_function(key)
        cache[key] = result
        return result

    return wrapper


@memoize
def function_using_cached_one_arg_func(x: int) -> int:
    f = function_using_global_variable
    f2 = cached_one_arg_func(f)
    return f2(x)  # type: ignore


def test_function_using_cached_one_arg_func() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert function_using_cached_one_arg_func(1) == 45

        # test caching works
        assert function_using_cached_one_arg_func(1) == 45
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None and last_cache_loading.endswith(".function_using_cached_one_arg_func")

        # test cache invalidation works
        reset_last_cache_loading()
        global global_a
        value_before = global_a
        global_a = 40
        assert function_using_cached_one_arg_func(1) == 40
        assert get_last_cache_loading() is None
        global_a = value_before


if __name__ == "__main__":
    test_global_variable_dependency_ext()

    test_function_using_cached_one_arg_func()

    test_function_using_injection()
    test_function_using_alias()
    test_function_using_global_func_variable()

    test_global_variable_dependency()
    test_pure_function_buildin_functions()
