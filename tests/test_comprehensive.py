"""Comprehensive test suite for memodisk — happy-path and edge-case coverage.

Organized into sections matching the memodisk feature surface:
  1. Basic caching (cache hit/miss, argument variations)
  2. Return type diversity
  3. Argument type diversity
  4. Code dependency tracking
  5. Data dependency tracking (open-based)
  6. Global variable tracking
  7. Closure variable tracking
  8. Nested / recursive memoization
  9. Class and method memoization
  10. Cache management (set_cache_dir, file naming, corruption)
  11. Error handling (exceptions inside memoized functions)
  12. DataLoaderWrapper
  13. add_data_dependency
  14. user_ignore_files
  15. hashing_func_map
"""

import dataclasses
import enum
import glob
import json
import math
import os
import pathlib
import pickle
import tempfile
import time
from typing import Any, cast

import numpy as np
import pytest

from memodisk import (
    ArgumentHasher,
    DataLoaderWrapper,
    ResultSerializer,
    add_data_dependency,
    get_last_cache_loading,
    load_cached_call_arguments,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)

# ===================================================================
# Helpers
# ===================================================================

def _fresh_cache():
    """Context manager that sets up a temp cache dir."""
    return tempfile.TemporaryDirectory(prefix="memodisk_cache_tests")


def _assert_cached(name_substr: str = "") -> None:
    """Assert the last call was a cache hit, optionally matching name."""
    last = get_last_cache_loading()
    assert last is not None, "Expected cache hit"
    if name_substr:
        assert name_substr in last, f"Expected '{name_substr}' in '{last}'"


def _assert_not_cached() -> None:
    """Assert the last call was a cache miss."""
    assert get_last_cache_loading() is None, "Expected cache miss"


# ===================================================================
# 1. Basic caching: cache hit, miss, different args
# ===================================================================

def _pure_multiply(a: int, b: int) -> int:
    return a * b


@memoize
def basic_func(x: int) -> int:
    return x * x


@memoize
def two_arg_func(a: int, b: int) -> int:
    return _pure_multiply(a, b)


@memoize
def kwarg_func(a: int, b: int = 10) -> int:
    return a + b


@memoize
def no_arg_func() -> str:
    return "hello"


def _cache_even_inputs(x: int) -> bool:
    return x % 2 == 0


@memoize(condition=_cache_even_inputs)
def conditional_func(x: int) -> int:
    return x * 10


argument_hasher_events: list[str] = []


def _callable_bytecode_hash_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    argument_hasher_events.append("hash")
    assert kwargs == {}
    (func_arg,) = args
    return func_arg.__code__.co_code.hex()


CALLABLE_BYTECODE_HASHER = ArgumentHasher(
    name="callable-bytecode",
    hash_args=_callable_bytecode_hash_args,
)


@memoize(argument_hasher=CALLABLE_BYTECODE_HASHER)
def func_with_custom_argument_hasher(f: Any) -> str:
    return "cached"


@memoize(store_call_arguments=True)
def func_with_stored_call_arguments(a: int, b: str = "default") -> tuple[int, str]:
    return (a, b)


serializer_events: list[str] = []


def _json_result_dumps(value: dict[str, Any]) -> bytes:
    serializer_events.append("dump")
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _json_result_loads(data: bytes) -> dict[str, Any]:
    serializer_events.append("load")
    return json.loads(data.decode("utf-8"))


JSON_RESULT_SERIALIZER = ResultSerializer[dict[str, Any]](
    name="json",
    dumps=_json_result_dumps,
    loads=_json_result_loads,
)


@memoize(serializer=JSON_RESULT_SERIALIZER)
def json_result_func() -> dict[str, Any]:
    return {"nested": {"a": 1}, "numbers": [1, 2, 3]}


class TestBasicCaching:
    def test_first_call_is_miss(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            reset_last_cache_loading()
            basic_func(3)
            _assert_not_cached()

    def test_second_call_is_hit(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(3)
            reset_last_cache_loading()
            result = basic_func(3)
            assert result == 9
            _assert_cached("basic_func")

    def test_different_args_no_hit(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(3)
            reset_last_cache_loading()
            basic_func(4)
            _assert_not_cached()

    def test_many_different_args(self) -> None:
        """Each distinct argument set creates its own cache entry."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            for i in range(20):
                basic_func(i)
            # All should now be cached
            for i in range(20):
                reset_last_cache_loading()
                assert basic_func(i) == i * i
                _assert_cached()

    def test_two_args(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert two_arg_func(3, 4) == 12
            reset_last_cache_loading()
            assert two_arg_func(3, 4) == 12
            _assert_cached()

    def test_kwargs_default(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert kwarg_func(5) == 15
            reset_last_cache_loading()
            assert kwarg_func(5) == 15
            _assert_cached()

    def test_kwargs_explicit_same_as_default(self) -> None:
        """kwarg_func(5) and kwarg_func(5, b=10) may or may not share cache."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            r1 = kwarg_func(5)
            r2 = kwarg_func(5, b=10)
            assert r1 == r2 == 15

    def test_kwargs_different_value(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert kwarg_func(5, b=20) == 25
            reset_last_cache_loading()
            assert kwarg_func(5, b=20) == 25
            _assert_cached()

    def test_no_arg_function(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert no_arg_func() == "hello"
            reset_last_cache_loading()
            assert no_arg_func() == "hello"
            _assert_cached()

    def test_cache_produces_files(self) -> None:
        """Verify that .pkl and .json files are created."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(99)
            pkl_files = glob.glob(os.path.join(d, "*_result.pkl"))
            json_files = glob.glob(os.path.join(d, "*_dependencies.json"))
            assert len(pkl_files) >= 1
            assert len(json_files) >= 1

    def test_dependency_json_is_valid(self) -> None:
        """The dependencies JSON should be well-formed and have expected keys."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(42)
            json_files = glob.glob(os.path.join(d, "*_dependencies.json"))
            assert len(json_files) >= 1
            with open(json_files[0]) as f:
                deps = json.load(f)
            assert "arguments_hash" in deps
            assert "code" in deps
            assert "data" in deps
            assert "random_states" in deps
            assert isinstance(deps["code"], list)
            assert len(deps["code"]) > 0
            for code_dep in deps["code"]:
                assert "function_qualified_name" in code_dep
                assert "bytecode_hash" in code_dep
                assert "filename" in code_dep

    def test_condition_false_skips_cache(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)

            assert conditional_func(3) == 30
            reset_last_cache_loading()
            assert conditional_func(3) == 30

            _assert_not_cached()
            assert glob.glob(os.path.join(d, "*_result.pkl")) == []
            assert glob.glob(os.path.join(d, "*_dependencies.json")) == []

    def test_condition_true_uses_cache(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)

            assert conditional_func(4) == 40
            reset_last_cache_loading()
            assert conditional_func(4) == 40

            _assert_cached("conditional_func")

    def test_condition_mixed_inputs_only_cache_when_enabled(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)

            assert conditional_func(3) == 30
            assert glob.glob(os.path.join(d, "*_result.pkl")) == []
            assert glob.glob(os.path.join(d, "*_dependencies.json")) == []

            assert conditional_func(4) == 40
            assert len(glob.glob(os.path.join(d, "*_result.pkl"))) == 1
            assert len(glob.glob(os.path.join(d, "*_dependencies.json"))) == 1

            reset_last_cache_loading()
            assert conditional_func(3) == 30
            _assert_not_cached()

            reset_last_cache_loading()
            assert conditional_func(4) == 40
            _assert_cached("conditional_func")

    def test_custom_result_serializer_round_trips_cache(self) -> None:
        with _fresh_cache() as d:
            serializer_events.clear()
            set_cache_dir(d)

            assert json_result_func() == {"nested": {"a": 1}, "numbers": [1, 2, 3]}
            assert serializer_events == ["dump"]

            dep_files = glob.glob(os.path.join(d, "*_dependencies.json"))
            result_files = glob.glob(os.path.join(d, "*_result.pkl"))
            assert len(dep_files) == 1
            assert len(result_files) == 1

            with open(dep_files[0], encoding="utf-8") as fh:
                all_dependencies = json.load(fh)
            assert all_dependencies["result_serializer"] == "json"

            with open(result_files[0], "rb") as fh:
                assert fh.read().startswith(b'{"nested": {"a": 1}, "numbers": [1, 2, 3]}')

            serializer_events.clear()
            reset_last_cache_loading()
            assert json_result_func() == {"nested": {"a": 1}, "numbers": [1, 2, 3]}
            _assert_cached("json_result_func")
            assert serializer_events == ["load"]

    def test_custom_argument_hasher_supports_unpicklable_arguments(self) -> None:
        with _fresh_cache() as d:
            argument_hasher_events.clear()
            set_cache_dir(d)

            def fn(x: int) -> int:
                return x * 3

            assert func_with_custom_argument_hasher(fn) == "cached"
            assert argument_hasher_events == ["hash"]

            dep_files = glob.glob(os.path.join(d, "*_dependencies.json"))
            assert len(dep_files) == 1
            with open(dep_files[0], encoding="utf-8") as fh:
                all_dependencies = json.load(fh)
            assert all_dependencies["argument_hasher"] == "callable-bytecode"

            argument_hasher_events.clear()
            reset_last_cache_loading()
            assert func_with_custom_argument_hasher(fn) == "cached"
            _assert_cached("func_with_custom_argument_hasher")
            assert argument_hasher_events == ["hash"]

    def test_store_call_arguments_writes_replay_pickle(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)

            assert func_with_stored_call_arguments(7, b="hello") == (7, "hello")

            dep_files = glob.glob(os.path.join(d, "*_dependencies.json"))
            call_files = glob.glob(os.path.join(d, "*_call_arguments.pkl"))
            assert len(dep_files) == 1
            assert len(call_files) == 1

            with open(dep_files[0], encoding="utf-8") as fh:
                all_dependencies = json.load(fh)

            assert all_dependencies["store_call_arguments"] is True
            assert all_dependencies["call_arguments_file"] == call_files[0]

            with open(call_files[0], "rb") as fh:
                payload = pickle.load(fh)
            assert payload == {"args": (7,), "kwargs": {"b": "hello"}}
            assert load_cached_call_arguments(dep_files[0]) == payload

            reset_last_cache_loading()
            assert func_with_stored_call_arguments(7, b="hello") == (7, "hello")
            _assert_cached("func_with_stored_call_arguments")


# ===================================================================
# 2. Return type diversity
# ===================================================================

def _identity(x):
    return x


@memoize
def return_none() -> None:
    _identity(None)
    return None


@memoize
def return_dict() -> dict:
    return _identity({"a": 1, "b": [2, 3], "c": {"nested": True}})


@memoize
def return_list() -> list:
    return _identity([1, "two", 3.0, None, [4, 5]])


@memoize
def return_tuple() -> tuple:
    return _identity((1, 2, 3))


@memoize
def return_set() -> frozenset:
    return _identity(frozenset({1, 2, 3}))


@memoize
def return_bytes() -> bytes:
    return _identity(b"\x00\x01\x02\xff")


@memoize
def return_numpy_array() -> np.ndarray:
    return np.array([[1, 2], [3, 4]], dtype=np.float64)


@dataclasses.dataclass
class Point:
    x: float
    y: float


@memoize
def return_dataclass() -> Point:
    return _identity(Point(1.0, 2.0))


class Color(enum.Enum):
    RED = 1
    GREEN = 2


@memoize
def return_enum() -> Color:
    return _identity(Color.RED)


class TestReturnTypes:
    @pytest.mark.parametrize(
        "func, expected",
        [
            (return_none, None),
            (return_dict, {"a": 1, "b": [2, 3], "c": {"nested": True}}),
            (return_list, [1, "two", 3.0, None, [4, 5]]),
            (return_tuple, (1, 2, 3)),
            (return_set, frozenset({1, 2, 3})),
            (return_bytes, b"\x00\x01\x02\xff"),
            (return_dataclass, Point(1.0, 2.0)),
            (return_enum, Color.RED),
        ],
    )
    def test_return_type_roundtrip(self, func, expected) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            result1 = func()
            assert result1 == expected
            reset_last_cache_loading()
            result2 = func()
            assert result2 == expected
            _assert_cached()

    def test_numpy_array_roundtrip(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            r1 = return_numpy_array()
            reset_last_cache_loading()
            r2 = return_numpy_array()
            np.testing.assert_array_equal(r1, r2)
            _assert_cached()


# ===================================================================
# 3. Argument type diversity
# ===================================================================

@memoize
def echo_arg(x: Any) -> Any:
    return _identity(x)


class TestArgumentTypes:
    @pytest.mark.parametrize(
        "arg",
        [
            0,
            -1,
            2**63,
            3.14,
            float("inf"),
            True,
            False,
            "",
            "hello unicode: café ñ 日本語",
            b"bytes arg",
            None,
            (1, 2, 3),
            (1, (2, (3,))),
        ],
    )
    def test_various_arg_types(self, arg: Any) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            r1 = echo_arg(arg)
            assert r1 == arg
            reset_last_cache_loading()
            r2 = echo_arg(arg)
            assert r2 == arg
            _assert_cached()

    def test_numpy_array_arg(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            arr = np.array([1, 2, 3])
            r1 = echo_arg(arr)
            np.testing.assert_array_equal(r1, arr)
            reset_last_cache_loading()
            r2 = echo_arg(arr)
            np.testing.assert_array_equal(r2, arr)
            _assert_cached()

    def test_dict_arg(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            arg = {"key": [1, 2, 3]}
            r1 = echo_arg(arg)
            assert r1 == arg
            reset_last_cache_loading()
            r2 = echo_arg(arg)
            assert r2 == arg
            _assert_cached()

    def test_list_arg(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            arg = [1, "mixed", None, 3.14]
            r1 = echo_arg(arg)
            assert r1 == arg
            reset_last_cache_loading()
            r2 = echo_arg(arg)
            assert r2 == arg
            _assert_cached()

    def test_dataclass_arg(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            p = Point(1.0, 2.0)
            r1 = echo_arg(p)
            assert r1 == p
            reset_last_cache_loading()
            r2 = echo_arg(p)
            assert r2 == p
            _assert_cached()


# ===================================================================
# 4. Code dependency tracking
# ===================================================================

def helper_add(a: int, b: int) -> int:
    return a + b


def helper_mul(a: int, b: int) -> int:
    return a * b


@memoize
def func_chain(x: int) -> int:
    """Calls two helpers in sequence."""
    return helper_add(helper_mul(x, x), 1)


@memoize
def func_with_conditional(x: int) -> int:
    """Only one code path taken depending on x."""
    if x > 0:
        return helper_add(x, 1)
    else:
        return helper_mul(x, -1)


@memoize
def func_with_list_comprehension(n: int) -> list:
    return [helper_add(i, i) for i in range(n)]


@memoize
def func_with_builtin_only(x: float) -> int:
    """Uses only builtins (math.ceil imported locally)."""
    return math.ceil(x)


@memoize
def func_with_try_except(x: int) -> str:
    try:
        return str(helper_add(x, 1))
    except Exception:
        return "error"


class TestCodeDependencyTracking:
    def test_helper_chain_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_chain(5) == 26
            reset_last_cache_loading()
            assert func_chain(5) == 26
            _assert_cached()

    def test_conditional_only_tracks_executed_branch(self) -> None:
        """Positive path should not depend on helper_mul."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_conditional(5) == 6
            json_files = glob.glob(os.path.join(d, "*conditional*_dependencies.json"))
            assert len(json_files) == 1
            with open(json_files[0]) as f:
                deps = json.load(f)
            dep_names = [c["function_qualified_name"] for c in deps["code"]]
            assert any("helper_add" in n for n in dep_names)
            # helper_mul should NOT be in deps for x > 0
            assert not any("helper_mul" in n for n in dep_names)

    def test_list_comprehension_dep(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_list_comprehension(3) == [0, 2, 4]
            reset_last_cache_loading()
            assert func_with_list_comprehension(3) == [0, 2, 4]
            _assert_cached()

    def test_builtin_math_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_builtin_only(2.3) == 3
            reset_last_cache_loading()
            assert func_with_builtin_only(2.3) == 3
            _assert_cached()

    def test_try_except_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_try_except(5) == "6"
            reset_last_cache_loading()
            assert func_with_try_except(5) == "6"
            _assert_cached()

    def test_dependencies_json_lists_helpers(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            func_chain(7)
            json_files = glob.glob(os.path.join(d, "*func_chain*_dependencies.json"))
            assert len(json_files) == 1
            with open(json_files[0]) as f:
                deps = json.load(f)
            dep_names = [c["function_qualified_name"] for c in deps["code"]]
            assert any("helper_add" in n for n in dep_names)
            assert any("helper_mul" in n for n in dep_names)


# ===================================================================
# 5. Data dependency tracking (via builtin open)
# ===================================================================

@memoize
def read_file_content(path: str) -> str:
    with open(path) as f:
        return f.read()


@memoize
def read_binary_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


@memoize
def read_file_no_context_mgr(path: str) -> str:
    f = open(path)
    content = f.read()
    f.close()
    return content


@memoize
def read_multiple_files(p1: str, p2: str) -> str:
    with open(p1) as f1, open(p2) as f2:
        return f1.read() + "|" + f2.read()


class TestDataDependencyTracking:
    def _write(self, path: str, content: str) -> None:
        with open(path, "w") as f:
            f.write(content)

    def test_file_read_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "test.txt")
            self._write(fp, "content_v1")
            assert read_file_content(fp) == "content_v1"
            reset_last_cache_loading()
            assert read_file_content(fp) == "content_v1"
            _assert_cached()

    def test_file_change_invalidates_cache(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "test.txt")
            self._write(fp, "v1")
            assert read_file_content(fp) == "v1"

            # Modify file
            self._write(fp, "v2")
            reset_last_cache_loading()
            assert read_file_content(fp) == "v2"
            _assert_not_cached()

    def test_binary_file_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "data.bin")
            with open(fp, "wb") as f:
                f.write(b"\x00\x01\x02")
            assert read_binary_file(fp) == b"\x00\x01\x02"
            reset_last_cache_loading()
            assert read_binary_file(fp) == b"\x00\x01\x02"
            _assert_cached()

    def test_no_context_manager(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "test.txt")
            self._write(fp, "hello")
            assert read_file_no_context_mgr(fp) == "hello"
            reset_last_cache_loading()
            assert read_file_no_context_mgr(fp) == "hello"
            _assert_cached()

    def test_multiple_file_deps(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            f1 = os.path.join(d, "a.txt")
            f2 = os.path.join(d, "b.txt")
            self._write(f1, "A")
            self._write(f2, "B")
            assert read_multiple_files(f1, f2) == "A|B"
            reset_last_cache_loading()
            assert read_multiple_files(f1, f2) == "A|B"
            _assert_cached()

    def test_one_of_multiple_files_changed(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            f1 = os.path.join(d, "a.txt")
            f2 = os.path.join(d, "b.txt")
            self._write(f1, "A")
            self._write(f2, "B")
            read_multiple_files(f1, f2)

            # Only modify second file
            self._write(f2, "B2")
            reset_last_cache_loading()
            assert read_multiple_files(f1, f2) == "A|B2"
            _assert_not_cached()

    def test_data_dep_in_json(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "tracked.txt")
            self._write(fp, "data")
            read_file_content(fp)
            json_files = glob.glob(os.path.join(d, "*read_file_content*_dependencies.json"))
            assert len(json_files) >= 1
            with open(json_files[0]) as f:
                deps = json.load(f)
            data_paths = [e["file_path"] for e in deps["data"]]
            assert any("tracked.txt" in p for p in data_paths)


# ===================================================================
# 6. Global variable tracking
# ===================================================================

_global_scalar = 100


def _use_global_scalar(x: int) -> int:
    return x * _global_scalar


@memoize
def func_with_global_dep(x: int) -> int:
    return _use_global_scalar(x)


class TestGlobalVariableTracking:
    def test_global_change_invalidates(self) -> None:
        global _global_scalar
        original = _global_scalar
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_global_dep(5) == 500
            reset_last_cache_loading()
            assert func_with_global_dep(5) == 500
            _assert_cached()

            _global_scalar = 200
            reset_last_cache_loading()
            assert func_with_global_dep(5) == 1000
            _assert_not_cached()
            _global_scalar = original

    def test_global_change_and_recompute(self) -> None:
        """After global changes, a fresh call recomputes correctly."""
        global _global_scalar
        original = _global_scalar
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_global_dep(5) == 500

            _global_scalar = 200
            reset_last_cache_loading()
            result = func_with_global_dep(5)
            assert result == 1000
            _assert_not_cached()

            # Now this new value is cached
            reset_last_cache_loading()
            assert func_with_global_dep(5) == 1000
            _assert_cached()
            _global_scalar = original


# ===================================================================
# 7. Closure variable tracking
# ===================================================================

class TestClosureTracking:
    def test_closure_basic(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)

            def make_add(n: int):
                @memoize
                def add(x: int) -> int:
                    return x + n
                return add

            add5 = make_add(5)
            assert add5(10) == 15
            reset_last_cache_loading()
            assert add5(10) == 15
            _assert_cached()

    def test_different_closures_different_caches(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)

            def make_mul(n: int):
                @memoize
                def mul(x: int) -> int:
                    return x * n
                return mul

            mul3 = make_mul(3)
            mul7 = make_mul(7)
            assert mul3(10) == 30
            assert mul7(10) == 70


# ===================================================================
# 8. Nested / recursive memoization
# ===================================================================

@memoize
def outer(x: int) -> int:
    return inner(x) + 1


@memoize
def inner(x: int) -> int:
    return x * 2


@memoize
def recursive_fib(n: int) -> int:
    if n <= 1:
        return n
    return recursive_fib(n - 1) + recursive_fib(n - 2)


class TestNestedMemoization:
    def test_nested_both_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert outer(5) == 11  # inner(5)=10, +1
            reset_last_cache_loading()
            assert outer(5) == 11
            _assert_cached()

    def test_inner_cached_independently(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            outer(5)  # this also calls inner(5)
            reset_last_cache_loading()
            assert inner(5) == 10
            _assert_cached("inner")

    def test_recursive_memoization(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert recursive_fib(10) == 55
            # All sub-calls should be cached now
            reset_last_cache_loading()
            assert recursive_fib(10) == 55
            _assert_cached()


# ===================================================================
# 9. Class and method memoization
# ===================================================================

class Calculator:
    def __init__(self, base: int):
        self.base = base

    @memoize
    def compute(self, x: int) -> int:
        return self.base + x


class ClassWithCallable:
    def __init__(self, factor: int):
        self.factor = factor

    @memoize
    def __call__(self, x: int) -> int:
        return self.factor * x


class TestClassMemoization:
    def test_method_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            c = Calculator(10)
            assert c.compute(5) == 15
            reset_last_cache_loading()
            assert c.compute(5) == 15
            _assert_cached()

    def test_different_instances_different_results(self) -> None:
        """Different self → different pickle → different cache key."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            c1 = Calculator(10)
            c2 = Calculator(20)
            assert c1.compute(5) == 15
            assert c2.compute(5) == 25

    def test_callable_class_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            c = ClassWithCallable(3)
            assert c(7) == 21
            reset_last_cache_loading()
            assert c(7) == 21
            _assert_cached()


# ===================================================================
# 10. Cache management
# ===================================================================

class TestCacheManagement:
    def test_set_cache_dir_changes_location(self) -> None:
        with _fresh_cache() as d1, _fresh_cache() as d2:
            set_cache_dir(d1)
            basic_func(100)
            assert len(glob.glob(os.path.join(d1, "*.pkl"))) >= 1
            assert len(glob.glob(os.path.join(d2, "*.pkl"))) == 0

            set_cache_dir(d2)
            basic_func(100)
            assert len(glob.glob(os.path.join(d2, "*.pkl"))) >= 1

    def test_cache_files_named_with_function(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(42)
            files = os.listdir(d)
            assert any("basic_func" in f for f in files)

    def test_corrupted_pkl_triggers_recompute(self) -> None:
        """If the pickle file is corrupted, memodisk should recompute."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(77)

            # Corrupt the pkl file
            pkl_files = glob.glob(os.path.join(d, "*basic_func*_result.pkl"))
            assert len(pkl_files) >= 1
            with open(pkl_files[0], "wb") as f:
                f.write(b"corrupted data")

            reset_last_cache_loading()
            result = basic_func(77)
            assert result == 77 * 77
            # Should have recomputed (cache miss after corruption)

    def test_missing_pkl_triggers_recompute(self) -> None:
        """If pkl is deleted but json remains, should recompute."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(88)

            pkl_files = glob.glob(os.path.join(d, "*basic_func*_result.pkl"))
            for f in pkl_files:
                os.remove(f)

            reset_last_cache_loading()
            result = basic_func(88)
            assert result == 88 * 88


# ===================================================================
# 11. Error handling
# ===================================================================

def _raiser_helper(x: int) -> int:
    if x < 0:
        raise ValueError(f"negative: {x}")
    return x * 2


@memoize
def func_that_raises(x: int) -> int:
    return _raiser_helper(x)


def _try_helper(x: int) -> str:
    try:
        val = 10 // x
    except ZeroDivisionError:
        val = -1
    return str(val)


@memoize
def func_with_internal_try(x: int) -> str:
    return _try_helper(x)


class TestErrorHandling:
    def test_exception_not_cached(self) -> None:
        """If a memoized function raises, the result should NOT be cached."""
        with _fresh_cache() as d:
            set_cache_dir(d)

            with pytest.raises(ValueError):
                func_that_raises(-1)

            # Call again — should raise again (not serve from cache)
            with pytest.raises(ValueError):
                func_that_raises(-1)

    def test_success_after_error_caches(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            with pytest.raises(ValueError):
                func_that_raises(-5)
            # Now call with valid arg
            assert func_that_raises(5) == 10
            reset_last_cache_loading()
            assert func_that_raises(5) == 10
            _assert_cached()

    def test_internal_try_except_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            assert func_with_internal_try(0) == "-1"
            reset_last_cache_loading()
            assert func_with_internal_try(0) == "-1"
            _assert_cached()


# ===================================================================
# 12. DataLoaderWrapper
# ===================================================================

def fake_imread(path: str) -> str:
    """Simulates cv2.imread by returning file content as string."""
    with open(path) as f:  # use raw open to not double-count
        return f.read()


wrapped_imread = DataLoaderWrapper(fake_imread)


@memoize
def load_image(path: str) -> str:
    return wrapped_imread(path)


class TestDataLoaderWrapper:
    def test_wrapper_tracks_dependency(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "img.txt")
            with open(fp, "w") as f:
                f.write("pixels_v1")

            assert load_image(fp) == "pixels_v1"
            reset_last_cache_loading()
            assert load_image(fp) == "pixels_v1"
            _assert_cached()

    def test_wrapper_detects_change(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "img.txt")
            with open(fp, "w") as f:
                f.write("pixels_v1")
            load_image(fp)

            with open(fp, "w") as f:
                f.write("pixels_v2")
            reset_last_cache_loading()
            assert load_image(fp) == "pixels_v2"
            _assert_not_cached()


# ===================================================================
# 13. add_data_dependency (manual)
# ===================================================================

@memoize
def func_with_manual_dep(path: str) -> str:
    add_data_dependency(path)
    return pathlib.Path(path).read_text()


class TestAddDataDependency:
    def test_manual_dep_cached(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "manual.txt")
            pathlib.Path(fp).write_text("v1")
            assert func_with_manual_dep(fp) == "v1"
            reset_last_cache_loading()
            assert func_with_manual_dep(fp) == "v1"
            _assert_cached()

    def test_manual_dep_invalidated_on_change(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            fp = os.path.join(d, "manual.txt")
            pathlib.Path(fp).write_text("v1")
            func_with_manual_dep(fp)

            # Sleep to ensure mtime changes. In Python 3.12+, pathlib.Path.read_text()
            # uses io.open() directly (not builtins.open), bypassing the
            # memoized_open_wrapper's loop_until_access_time mechanism.
            time.sleep(0.05)
            pathlib.Path(fp).write_text("v2")
            reset_last_cache_loading()
            assert func_with_manual_dep(fp) == "v2"
            _assert_not_cached()


# ===================================================================
# 14. user_ignore_files
# ===================================================================

class TestUserIgnoreFiles:
    def test_user_ignore_files_is_configurable(self) -> None:
        """user_ignore_files is a set we can add to / remove from."""
        from memodisk import user_ignore_files
        assert isinstance(user_ignore_files, set)
        user_ignore_files.add("some_fake_path.py")
        assert "some_fake_path.py" in user_ignore_files
        user_ignore_files.discard("some_fake_path.py")
        assert "some_fake_path.py" not in user_ignore_files


# ===================================================================
# 15. hashing_func_map
# ===================================================================

class TestHashingFuncMap:
    def test_hashing_func_map_is_extensible(self) -> None:
        """hashing_func_map allows registering custom hash functions."""
        from memodisk import hashing_func_map
        assert isinstance(hashing_func_map, dict)
        # Can register and unregister
        sentinel = object()
        hashing_func_map[type(sentinel)] = lambda _value: "fixed"
        assert type(sentinel) in hashing_func_map
        del hashing_func_map[type(sentinel)]
        assert type(sentinel) not in hashing_func_map


# ===================================================================
# 16. Numpy random state
# ===================================================================

@memoize
def func_with_random() -> float:
    return float(np.random.rand())


class TestNumpyRandomState:
    def test_same_seed_same_result(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            np.random.seed(42)
            r1 = func_with_random()
            np.random.seed(42)
            reset_last_cache_loading()
            r2 = func_with_random()
            assert r1 == r2
            _assert_cached()

    def test_different_seed_different_result(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            np.random.seed(42)
            r1 = func_with_random()
            np.random.seed(99)
            reset_last_cache_loading()
            r2 = func_with_random()
            assert r1 != r2
            _assert_not_cached()

    def test_random_state_restored_after_cache_hit(self) -> None:
        """After a cache hit, the RNG state should be as if the function ran."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            np.random.seed(0)
            func_with_random()
            state_after_first = cast(tuple[Any, ...], np.random.get_state())[1].copy()

            np.random.seed(0)
            func_with_random()  # cache hit
            state_after_second = cast(tuple[Any, ...], np.random.get_state())[1].copy()

            np.testing.assert_array_equal(state_after_first, state_after_second)


# ===================================================================
# 17. Comment and formatting changes should NOT invalidate
# ===================================================================

class TestBytecodeStability:
    def test_function_result_survives_comment_change(self) -> None:
        """Since memodisk hashes bytecode, adding/removing comments
        should not invalidate the cache. We test this indirectly:
        the same function object cached twice should always hit."""
        with _fresh_cache() as d:
            set_cache_dir(d)
            basic_func(123)
            reset_last_cache_loading()
            basic_func(123)
            _assert_cached()


# ===================================================================
# 18. Large argument / result
# ===================================================================

@memoize
def sum_large_array(arr: np.ndarray) -> float:
    return float(np.sum(arr))


class TestLargeData:
    def test_large_numpy_arg_and_result(self) -> None:
        with _fresh_cache() as d:
            set_cache_dir(d)
            big = np.ones(1_000_000, dtype=np.float64)
            r1 = sum_large_array(big)
            assert r1 == 1_000_000.0
            reset_last_cache_loading()
            r2 = sum_large_array(big)
            assert r2 == 1_000_000.0
            _assert_cached()
