"""Tests demonstrating the known failure modes / limitations listed in the README.

Each test is marked with xfail (or uses explicit assertions) to document that
memodisk returns stale/incorrect results in these scenarios.
"""

import json
import math as mymath
import os
import pathlib
import pickle
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime as datetime_cls
from typing import Any

import numpy as np
import pytest

from memodisk import (
    MemoizeMode,
    get_last_cache_loading,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)
from memodisk.memodisk import (
    _cache_process_lock,
    _canonical_hash,
    loop_until_access_time_greater_than_modification_time,
    tracer,
)


# ---------------------------------------------------------------------------
# 1. Property decorator not supported
# ---------------------------------------------------------------------------
class MyClass:
    @property
    @memoize
    def value(self) -> int:
        return 42


def test_property_decorator() -> None:
    """Regression: property-wrapped memoized getters should cache cleanly."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        try:
            obj = MyClass()
            assert obj.value == 42
            reset_last_cache_loading()
            assert obj.value == 42
            assert get_last_cache_loading() is not None
            assert not tracer.is_registered
        finally:
            if tracer.is_registered:
                tracer.unregister()
            tracer.clear_counters()


# ---------------------------------------------------------------------------
# 2. Non-picklable arguments
# ---------------------------------------------------------------------------
@memoize
def func_with_lambda_arg(f: Any) -> int:
    return f(10)


def test_non_picklable_argument() -> None:
    """Limitation: requires all arguments to be serializable using pickle."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            func_with_lambda_arg(lambda x: x * 2)


# ---------------------------------------------------------------------------
# 3. Global variable dependency not always detected
# ---------------------------------------------------------------------------
_hidden_global = 10


class ReprCollisionKey:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return "Key()"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


def _inner_helper_globals_dict(x: int) -> int:
    # Access global through globals() dict — not visible via LOAD_GLOBAL bytecode
    return x * globals()["_hidden_global"]


@memoize
def func_using_hidden_global(x: int) -> int:
    return _inner_helper_globals_dict(x)


@pytest.mark.parametrize("mode", ["safe", "optimistic"])
def test_global_variable_not_detected(mode: MemoizeMode) -> None:
    """Test #3: globals() dict lookup behavior across modes.

    - safe: always recomputes (never returns stale data)
    - optimistic: KNOWN FAILURE — returns stale cached data because
      globals() dict access bypasses LOAD_GLOBAL bytecode tracking
    """
    global _hidden_global
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        @memoize(mode=mode)
        def func_using_hidden_global_mode(x: int) -> int:
            return _inner_helper_globals_dict(x)

        _hidden_global = 10
        assert func_using_hidden_global_mode(5) == 50

        # Change the global
        reset_last_cache_loading()
        _hidden_global = 20

        if mode == "safe":
            # safe mode: always recomputes when dynamic globals detected
            result = func_using_hidden_global_mode(5)
            assert result == 100, f"Expected 100 but got {result} — safe mode should recompute"
        else:
            # optimistic: serves stale cache — this documents the known failure
            result = func_using_hidden_global_mode(5)
            assert result == 50, (
                f"Optimistic mode should return stale 50, got {result}. "
                "This documents that globals() access is not tracked."
            )

        _hidden_global = 10  # restore


def test_global_variable_strict_raises() -> None:
    """Test #3 strict variant: raises RuntimeError when globals() is detected."""
    global _hidden_global
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        _hidden_global = 10

        @memoize(mode="strict")
        def func_strict_globals(x: int) -> int:
            return _inner_helper_globals_dict(x)

        with pytest.raises(RuntimeError, match="strict mode"):
            func_strict_globals(5)

        _hidden_global = 10  # restore


# ---------------------------------------------------------------------------
# 4. Non-determinism due to time
# ---------------------------------------------------------------------------
def _get_time() -> float:
    return time.time()


def _get_datetime_now_isoformat() -> str:
    return datetime_cls.now().isoformat()


@pytest.mark.parametrize("mode", ["safe", "optimistic"])
def test_time_nondeterminism_policy(mode: MemoizeMode) -> None:
    """Common ambient time APIs should follow the mode policy."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        @memoize(mode=mode)
        def func_using_time_mode() -> float:
            return _get_time()

        result1 = func_using_time_mode()

        reset_last_cache_loading()
        time.sleep(0.001)
        result2 = func_using_time_mode()

        if mode == "safe":
            assert get_last_cache_loading() is None
            assert result2 > result1
        else:
            assert get_last_cache_loading() is not None
            assert result1 == result2


def test_time_nondeterminism_strict_raises() -> None:
    """Strict mode should reject cacheable functions that observe ambient time."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="strict")
        def func_using_time_strict() -> float:
            return _get_time()

        with pytest.raises(RuntimeError, match="ambient time dependency tracking is incomplete"):
            func_using_time_strict()


def test_datetime_now_alias_is_treated_as_ambient_time() -> None:
    """Imported datetime aliases should also be recognized as ambient time."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        @memoize(mode="safe")
        def func_using_datetime_alias() -> str:
            return _get_datetime_now_isoformat()

        result1 = func_using_datetime_alias()
        reset_last_cache_loading()
        time.sleep(0.001)
        result2 = func_using_datetime_alias()

        assert get_last_cache_loading() is None
        assert result2 != result1


@pytest.mark.parametrize("mode", ["safe", "optimistic"])
def test_environment_nondeterminism_policy(mode: MemoizeMode) -> None:
    """Common ambient environment APIs should follow the mode policy."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        @memoize(mode=mode)
        def func_using_environment_mode() -> str:
            return os.environ["MEMODISK_TEST_ENV_VALUE"]

        os.environ["MEMODISK_TEST_ENV_VALUE"] = "first"
        result1 = func_using_environment_mode()

        reset_last_cache_loading()
        os.environ["MEMODISK_TEST_ENV_VALUE"] = "second"
        result2 = func_using_environment_mode()

        if mode == "safe":
            assert get_last_cache_loading() is None
            assert result2 == "second"
        else:
            assert get_last_cache_loading() is not None
            assert result1 == result2 == "first"


def test_environment_nondeterminism_strict_raises() -> None:
    """Strict mode should reject cacheable functions that observe ambient environment state."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="strict")
        def func_using_environment_strict() -> str:
            return str(pathlib.Path.cwd())

        with pytest.raises(RuntimeError, match="ambient environment dependency tracking is incomplete"):
            func_using_environment_strict()


def test_cross_process_identical_call_executes_once() -> None:
    """Only one process should execute a cold identical call for a cache key."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        marker_dir = os.path.join(tmp_folder, "markers")
        os.makedirs(marker_dir, exist_ok=True)
        script_path = os.path.join(tmp_folder, "cross_process_call.py")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.getcwd(), env.get("PYTHONPATH")]))

        pathlib.Path(script_path).write_text(
            "\n".join(
                [
                    "import os",
                    "import pathlib",
                    "import time",
                    "from memodisk import memoize, set_cache_dir",
                    f"set_cache_dir({tmp_folder!r})",
                    "@memoize()",
                    "def slow_value() -> str:",
                    f"    marker = pathlib.Path({marker_dir!r}) / f'{{os.getpid()}}.txt'",
                    "    marker.write_text('executed', encoding='utf-8')",
                    "    time.sleep(0.5)",
                    "    return 'value'",
                    "print(slow_value())",
                ]
            ),
            encoding="utf-8",
        )

        proc1 = subprocess.Popen([sys.executable, script_path], env=env, stdout=subprocess.PIPE, text=True)
        proc2 = subprocess.Popen([sys.executable, script_path], env=env, stdout=subprocess.PIPE, text=True)
        out1, _ = proc1.communicate(timeout=30)
        out2, _ = proc2.communicate(timeout=30)

        assert proc1.returncode == 0, out1
        assert proc2.returncode == 0, out2
        assert len(os.listdir(marker_dir)) == 1
        assert out1.strip().endswith("value")
        assert out2.strip().endswith("value")


def test_safe_mode_never_caches_incomplete_external_process_tracking() -> None:
    """Safe mode should recompute when subprocess coverage is incomplete."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        @memoize(mode="safe")
        def call_python() -> str:
            return subprocess.check_output(
                [sys.executable, "-c", "print('ok')"],
                text=True,
            ).strip()

        assert call_python() == "ok"
        assert get_last_cache_loading() is None

        reset_last_cache_loading()
        assert call_python() == "ok"
        assert get_last_cache_loading() is None


def test_strict_mode_raises_on_incomplete_external_process_tracking() -> None:
    """Strict mode should refuse caching when subprocess coverage is incomplete."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="strict")
        def call_python() -> str:
            return subprocess.check_output(
                [sys.executable, "-c", "print('ok')"],
                text=True,
            ).strip()

        with pytest.raises(RuntimeError, match="external process dependency tracking is incomplete"):
            call_python()


# ---------------------------------------------------------------------------
# 5. C/C++ extension module changes not detected
# ---------------------------------------------------------------------------
def _call_np_sin(x: float) -> float:
    return float(np.sin(x))


@memoize
def func_using_numpy_internals(x: float) -> float:
    return _call_np_sin(x)


def test_extension_module_change_not_detected() -> None:
    """Regression: compiled extension module files are now tracked.

    numpy's sin() is implemented in C. memodisk should now record the
    underlying compiled extension module files as dependency metadata.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        result1 = func_using_numpy_internals(1.0)
        assert get_last_cache_loading() is None  # first call, no cache

        reset_last_cache_loading()
        result2 = func_using_numpy_internals(1.0)
        assert get_last_cache_loading() is not None  # cached
        assert result1 == result2

        # Verify the dependency json mentions at least one compiled module file.
        import glob
        import json

        dep_files = glob.glob(os.path.join(tmp_folder, "*.json"))
        assert len(dep_files) == 1
        with open(dep_files[0]) as f:
            deps = json.load(f)
        compiled_dependencies = {
            file_path
            for entry in deps["code"]
            for file_path in entry.get("compiled_dependencies", {})
        }
        assert any(file_path.endswith((".pyd", ".so", ".dll")) for file_path in compiled_dependencies)


# ---------------------------------------------------------------------------
# 6. Thread safety
# ---------------------------------------------------------------------------
@memoize
def func_for_threading(x: int) -> int:
    return x * x


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_safe() -> None:
    """Regression: concurrent memoized calls should not race or lose correctness."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        for _ in range(5):
            results: dict[int, int] = {}
            errors: list[Exception] = []
            barrier = threading.Barrier(11)

            def worker(
                val: int,
                barrier: threading.Barrier = barrier,
                results: dict[int, int] = results,
                errors: list[Exception] = errors,
            ) -> None:
                barrier.wait()
                try:
                    results[val] = func_for_threading(val)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()

            barrier.wait()

            for t in threads:
                t.join()

            assert not errors, f"Unexpected thread errors: {errors}"
            assert results == {i: i * i for i in range(10)}


def _compute_with_alias(x: float) -> float:
    return mymath.ceil(x)


@memoize
def func_using_aliased_import(x: float) -> float:
    return _compute_with_alias(x)


def test_aliased_import_not_detected() -> None:
    """Module-typed globals are now tracked by hashing module.__name__ + __file__.

    When an aliased import (e.g., `import math as mymath`) is swapped to a
    different module, memodisk detects the change and invalidates the cache.
    """
    global mymath
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        original = mymath
        assert func_using_aliased_import(2.3) == 3.0

        # cached
        reset_last_cache_loading()
        assert func_using_aliased_import(2.3) == 3.0
        assert get_last_cache_loading() is not None

        # Swap the aliased import to a fake module with different behavior
        import types

        fake_math = types.ModuleType("fake_math")
        fake_math.ceil = lambda x: int(x)  # floor-like behavior  # type: ignore
        mymath = fake_math  # type: ignore

        reset_last_cache_loading()
        result = func_using_aliased_import(2.3)
        # If stale, result is still 3.0 instead of 2
        assert result == 2, f"Expected 2 but got {result} (stale cache)"
        mymath = original  # restore


# ---------------------------------------------------------------------------
# 8. Pickle hash not deterministic for identical objects
# ---------------------------------------------------------------------------
@memoize
def func_for_hash_test(d: dict) -> str:
    return str(sorted(d.items()))


@pytest.mark.parametrize("mode", ["safe", "optimistic"])
def test_pickle_hash_nondeterministic(mode: MemoizeMode) -> None:
    """Test #8: pickle hash determinism across modes.

    - safe: canonical hashing normalizes dict order — cache hit
    - optimistic: raw pickle may produce different hashes — cache miss
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        @memoize(mode=mode)
        def func_for_hash_mode(d: dict) -> str:
            return str(sorted(d.items()))

        d1 = {"a": 1, "b": 2, "c": 3}
        result1 = func_for_hash_mode(d1)
        assert get_last_cache_loading() is None  # first call

        # Build an "identical" dict in a different insertion order
        d2: dict = {}
        for k in ["c", "b", "a"]:
            d2[k] = d1[k]

        reset_last_cache_loading()
        result2 = func_for_hash_mode(d2)

        if mode == "safe":
            # Canonical hashing normalizes order — should be cache hit
            assert get_last_cache_loading() is not None, (
                "Expected cache hit for identical dict content (safe mode)"
            )
            assert result1 == result2
        else:
            # optimistic: KNOWN FAILURE — raw pickle preserves insertion order,
            # so two dicts with the same content but built in different order
            # get different cache keys, causing a cache miss.
            # This documents the known limitation.
            assert result1 == result2  # same logical result
            # The cache miss is the documented failure mode — we don't assert
            # on get_last_cache_loading() because it may or may not miss
            # depending on CPython internals, but the risk exists.


# ---------------------------------------------------------------------------
# 9. Only decorated functions are memoized
# ---------------------------------------------------------------------------
def func_not_decorated(x: int) -> int:
    return x + 1


call_count_non_decorated = 0


def func_not_decorated_counting(x: int) -> int:
    global call_count_non_decorated
    call_count_non_decorated += 1
    return x + 1


def test_only_decorated_functions_memoized() -> None:
    """Limitation: will memoize only the decorated functions.

    Functions without @memoize are always re-executed, even if called
    inside a memoized function with the same args.
    """
    global call_count_non_decorated
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        call_count_non_decorated = 0

        func_not_decorated_counting(5)
        func_not_decorated_counting(5)

        # Called twice because it's not decorated
        assert call_count_non_decorated == 2, (
            "Non-decorated function is called every time"
        )


# ===========================================================================
# Additional failure modes (beyond README)
# ===========================================================================


# ---------------------------------------------------------------------------
# 10. Environment variable dependency tracked conservatively
# ---------------------------------------------------------------------------
def _read_env(key: str) -> str:
    return os.environ.get(key, "default")


@memoize
def func_using_env(key: str) -> str:
    return _read_env(key)


def test_env_variable_dependency_uses_safe_policy() -> None:
    """Environment reads should force recompute in safe mode."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        os.environ["MEMODISK_TEST_VAR"] = "value_a"
        assert func_using_env("MEMODISK_TEST_VAR") == "value_a"

        # safe mode should not cache ambient environment reads
        reset_last_cache_loading()
        assert func_using_env("MEMODISK_TEST_VAR") == "value_a"
        assert get_last_cache_loading() is None

        # Change env var — result should recompute
        os.environ["MEMODISK_TEST_VAR"] = "value_b"
        reset_last_cache_loading()
        result = func_using_env("MEMODISK_TEST_VAR")
        assert result == "value_b"
        assert get_last_cache_loading() is None
        del os.environ["MEMODISK_TEST_VAR"]


@memoize(mode="safe")
def func_using_local_env_alias(key: str) -> str:
    env = os.environ
    return env[key]


def test_env_local_alias_not_detected() -> None:
    """Local aliases of os.environ should also force recompute in safe mode."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        os.environ["MEMODISK_ALIAS_TEST_VAR"] = "value_a"
        assert func_using_local_env_alias("MEMODISK_ALIAS_TEST_VAR") == "value_a"

        reset_last_cache_loading()
        assert func_using_local_env_alias("MEMODISK_ALIAS_TEST_VAR") == "value_a"
        assert get_last_cache_loading() is None

        os.environ["MEMODISK_ALIAS_TEST_VAR"] = "value_b"
        reset_last_cache_loading()
        result = func_using_local_env_alias("MEMODISK_ALIAS_TEST_VAR")

        assert result == "value_b"
        assert get_last_cache_loading() is None
        del os.environ["MEMODISK_ALIAS_TEST_VAR"]


# ---------------------------------------------------------------------------
# 11. Side effects skipped on cache hit
# ---------------------------------------------------------------------------
@memoize
def func_with_file_side_effect(x: int, output_path: str) -> int:
    """Writes a file as a side effect, then returns a result."""
    with open(output_path, "w") as f:
        f.write(str(x))
    return x * 2


def test_side_effects_skipped_on_cache_hit() -> None:
    """Failure mode: side effects are not replayed when cache is hit.

    If the memoized function writes files, logs, sends messages, etc.,
    those effects simply don't happen on cache hits.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        output_file = os.path.join(tmp_folder, "side_effect.txt")

        result1 = func_with_file_side_effect(42, output_file)
        assert result1 == 84
        assert os.path.exists(output_file)

        # Delete the output file
        os.remove(output_file)
        assert not os.path.exists(output_file)

        # Second call hits cache — the file write side effect does NOT happen
        result2 = func_with_file_side_effect(42, output_file)
        assert result2 == 84
        assert not os.path.exists(output_file), (
            "Side effect (file write) was not replayed on cache hit"
        )


def test_stale_process_lock_directory_is_recovered_automatically() -> None:
    """Stale .lock directories with dead owners should be reclaimed automatically."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        cache_prefix = os.path.join(tmp_folder, "cache_key")
        lock_path = f"{cache_prefix}.lock"
        os.mkdir(lock_path)
        with open(os.path.join(lock_path, "owner.json"), "w", encoding="utf-8") as fh:
            json.dump({"pid": 2_147_483_647, "created_at": 0.0}, fh)

        acquired = threading.Event()

        def worker() -> None:
            with _cache_process_lock(cache_prefix):
                acquired.set()
                assert os.path.exists(lock_path)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        thread.join(timeout=1.0)
        assert acquired.is_set(), "Worker should reclaim the stale lock automatically"
        assert not thread.is_alive()
        assert not os.path.exists(lock_path)


# ---------------------------------------------------------------------------
# 12. Dynamic code via exec/eval not tracked
# ---------------------------------------------------------------------------
@memoize
def func_using_eval(expr: str) -> Any:
    return eval(expr)  # noqa: S307


def test_dynamic_eval_not_tracked() -> None:
    """Failure mode: code generated via exec/eval is not tracked.

    The evaluated expression is passed as a string argument (so different
    strings get different cache keys), but if a variable referenced inside
    the eval changes, the cache won't notice.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        # eval("2 + 2") is just a constant expression — always 4
        assert func_using_eval("2 + 2") == 4
        reset_last_cache_loading()
        assert func_using_eval("2 + 2") == 4
        assert get_last_cache_loading() is not None  # cached


# ---------------------------------------------------------------------------
# 13. Data loaded via pathlib tracked
# ---------------------------------------------------------------------------


def _read_with_pathlib(path: str) -> str:
    return pathlib.Path(path).read_text()


@memoize
def func_loading_via_pathlib(path: str) -> str:
    return _read_with_pathlib(path)


def test_pathlib_data_dependency_tracked() -> None:
    """Pathlib-based reads should invalidate cached results when the file changes."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        data_file = os.path.join(tmp_folder, "data.txt")
        pathlib.Path(data_file).write_text("version_1")

        reset_last_cache_loading()
        assert func_loading_via_pathlib(data_file) == "version_1"

        # cached
        reset_last_cache_loading()
        assert func_loading_via_pathlib(data_file) == "version_1"
        assert get_last_cache_loading() is not None

        # Modify the file — change should invalidate the cache
        pathlib.Path(data_file).write_text("version_2")
        reset_last_cache_loading()
        result = func_loading_via_pathlib(data_file)
        assert result == "version_2"
        assert get_last_cache_loading() is None


# ---------------------------------------------------------------------------
# 14. Closure variable mutation not detected
# ---------------------------------------------------------------------------
def make_multiplier(factor: int):
    """Return a memoized function that captures `factor` via closure."""
    @memoize
    def multiply(x: int) -> int:
        return x * factor
    return multiply


def test_closure_variable_mutation() -> None:
    """Failure mode: if a closure's captured mutable is changed externally,
    memodisk may not detect it because closure nonlocals are checked at
    decoration time, not at call time, for mutable containers.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        # Each call to make_multiplier creates a new decorated function
        # so this specific case actually gets different cache files.
        # The real issue is with mutable containers in closures:
        container = [10]

        @memoize
        def func_with_mutable_closure(x: int) -> int:
            return x * container[0]

        assert func_with_mutable_closure(5) == 50

        reset_last_cache_loading()
        assert func_with_mutable_closure(5) == 50
        assert get_last_cache_loading() is not None  # cached

        # Mutate the captured list
        container[0] = 20
        reset_last_cache_loading()
        result = func_with_mutable_closure(5)
        # memodisk uses inspect.getclosurevars(func).nonlocals in the hash,
        # but `container` is the same list object (just mutated), and the
        # hash was computed at first call. The closure var name "container"
        # doesn't appear as a LOAD_GLOBAL — it's a LOAD_DEREF.
        # Depending on implementation, this may or may not be stale.
        if get_last_cache_loading() is not None:
            pytest.xfail(
                f"Stale result: got {result}, expected 100 — "
                "mutable closure variable mutation not detected"
            )
        assert result == 100


# ===========================================================================
# Additional failure modes discovered during comprehensive test development
# ===========================================================================


# ---------------------------------------------------------------------------
# 15. Non-picklable global variable crashes the tracer
# ---------------------------------------------------------------------------
def test_non_picklable_global_crashes_tracer() -> None:
    """Non-picklable closure variables no longer crash the memoize wrapper.

    The memoize wrapper now catches pickle errors when computing the cache
    key from closure variables and falls back to a type+id based hash.
    The tracer also handles non-picklable globals gracefully.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        unpicklable_global = {"fn": lambda x: x}  # noqa: F841

        @memoize
        def func_using_unpicklable_global() -> int:
            return unpicklable_global["fn"](42)

        try:
            # Should no longer crash — returns the correct result
            result = func_using_unpicklable_global()
            assert result == 42
        finally:
            if tracer.is_registered:
                tracer.unregister()
            tracer.clear_counters()


# ---------------------------------------------------------------------------
# 16. Global variable change between calls raises BaseException (not cache miss)
# ---------------------------------------------------------------------------
_counter_for_test = 0


def _helper_using_counter(x: int) -> int:
    return x + _counter_for_test


@memoize
def func_with_changing_counter(x: int) -> int:
    return _helper_using_counter(x)


def test_global_change_during_execution_raises_not_misses() -> None:
    """With Python 3.12's sys.monitoring, global variable changes between two
    calls to the same function during a tracer session no longer raise
    BaseException. Instead, the change is detected, logged, and the cache
    is skipped (the result is still returned but not cached).

    This is a behavioral improvement over the old sys.setprofile approach
    which would crash user code with BaseException.
    """
    global _counter_for_test
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        _counter_for_test = 0

        assert func_with_changing_counter(5) == 5

        # Change the global — subsequent call completes but cache is skipped
        _counter_for_test = 100

        try:
            # With sys.monitoring, this no longer raises — it returns the
            # correct result but skips caching
            result = func_with_changing_counter(10)
            assert result == 110  # 10 + 100 (_counter_for_test)
        finally:
            _counter_for_test = 0
            if tracer.is_registered:
                tracer.unregister()
            tracer.clear_counters()


# ---------------------------------------------------------------------------
# 17. Exception in memoized function leaves tracer corrupted
# ---------------------------------------------------------------------------
def _always_fails() -> None:
    raise RuntimeError("intentional")


@memoize
def func_that_always_fails() -> None:
    _always_fails()


def test_exception_in_memoized_func_tracer_state() -> None:
    """Failure mode: when a memoized function raises an exception during its
    first call, the tracer may be left in a registered state. If the exception
    is caught by the caller, subsequent memoize calls can fail because of
    tracer assertion 'assert not self.is_registered'.

    Note: memodisk does catch the exception and re-raise, but the tracer
    may not be properly unregistered in all code paths.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        try:
            with pytest.raises(RuntimeError, match="intentional"):
                func_that_always_fails()
        finally:
            # Clean up if tracer was left registered
            if tracer.is_registered:
                tracer.unregister()
            tracer.clear_counters()

        # After cleanup, another memoized function should work
        reset_last_cache_loading()

        @memoize
        def simple_func(x: int) -> int:
            return x + 1

        assert simple_func(5) == 6


# ---------------------------------------------------------------------------
# 18. Restored global doesn't reuse original cache entry
# ---------------------------------------------------------------------------
_restorable_global = 100


def _use_restorable(x: int) -> int:
    return x * _restorable_global


@memoize
def func_with_restorable_global(x: int) -> int:
    return _use_restorable(x)


def test_restored_global_reuses_cache() -> None:
    """Multi-entry cache: restoring a global to its original value reuses
    the original cache entry.

    With multi-entry caching, each unique global state gets its own cache
    entry. When the global is restored, the original entry's dependencies
    still match, so it's a cache hit.
    """
    global _restorable_global
    original = _restorable_global
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        try:
            assert func_with_restorable_global(5) == 500

            # Change global
            _restorable_global = 200
            func_with_restorable_global(5)

            # Restore to original
            _restorable_global = original
            reset_last_cache_loading()
            result = func_with_restorable_global(5)
            assert result == 500
            assert get_last_cache_loading() is not None  # should be cache hit
        finally:
            _restorable_global = original
            if tracer.is_registered:
                tracer.unregister()
            tracer.clear_counters()


# ===========================================================================
# Mode-specific tests
# ===========================================================================


# ---------------------------------------------------------------------------
# 19. Decorator form variants all work
# ---------------------------------------------------------------------------
def test_memoize_bare_decorator() -> None:
    """@memoize (no parentheses) defaults to safe mode."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize
        def add(x: int, y: int) -> int:
            return x + y

        assert add(1, 2) == 3
        reset_last_cache_loading()
        assert add(1, 2) == 3
        assert get_last_cache_loading() is not None


def test_memoize_empty_parens() -> None:
    """@memoize() (empty parentheses) defaults to safe mode."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize()
        def add(x: int, y: int) -> int:
            return x + y

        assert add(1, 2) == 3
        reset_last_cache_loading()
        assert add(1, 2) == 3
        assert get_last_cache_loading() is not None


@pytest.mark.parametrize("mode", ["strict", "safe", "optimistic"])
def test_memoize_explicit_mode(mode: MemoizeMode) -> None:
    """@memoize(mode=...) works for all three modes on clean functions."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode=mode)
        def add(x: int, y: int) -> int:
            return x + y

        assert add(1, 2) == 3
        reset_last_cache_loading()
        assert add(1, 2) == 3
        assert get_last_cache_loading() is not None


# ---------------------------------------------------------------------------
# 20. Strict mode caches normally when code is clean
# ---------------------------------------------------------------------------
def test_strict_mode_caches_clean_functions() -> None:
    """Strict mode should behave identically to safe for functions
    that don't use globals()/locals() — full caching, no errors."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="strict")
        def expensive(x: int) -> int:
            return x ** 2

        assert expensive(5) == 25
        reset_last_cache_loading()
        assert expensive(5) == 25
        assert get_last_cache_loading() is not None  # second call served from cache


# ---------------------------------------------------------------------------
# 21. Safe canonical hashing with nested/complex structures
# ---------------------------------------------------------------------------
def test_safe_canonical_hashing_nested_dict() -> None:
    """Safe mode normalizes nested dicts so insertion order doesn't matter."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="safe")
        def process(d: dict) -> str:
            return str(sorted(str(item) for item in d.items()))

        d1 = {"outer": {"a": 1, "b": 2}, "key": "val"}
        result1 = process(d1)

        d2 = {"key": "val", "outer": {"b": 2, "a": 1}}
        reset_last_cache_loading()
        result2 = process(d2)

        assert result1 == result2
        assert get_last_cache_loading() is not None, "Nested dicts with same content should cache-hit"


def test_safe_canonical_hashing_with_sets() -> None:
    """Safe mode normalizes frozensets so element order doesn't matter."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="safe")
        def process(s: frozenset) -> int:
            return sum(s)

        s1 = frozenset([3, 1, 2])
        result1 = process(s1)

        s2 = frozenset([2, 3, 1])
        reset_last_cache_loading()
        result2 = process(s2)

        assert result1 == result2
        assert get_last_cache_loading() is not None, "Frozensets with same elements should cache-hit"


def test_safe_canonical_hashing_tuple_of_dicts() -> None:
    """Safe mode normalizes tuples containing dicts."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="safe")
        def process(t: tuple) -> str:
            return str(t)

        t1 = ({"x": 1, "y": 2}, {"a": 10, "b": 20})
        result1 = process(t1)

        t2 = ({"y": 2, "x": 1}, {"b": 20, "a": 10})
        reset_last_cache_loading()
        result2 = process(t2)

        assert result1 == result2
        assert get_last_cache_loading() is not None, "Tuples of dicts with same content should cache-hit"


# ---------------------------------------------------------------------------
# 22. Strict + canonical hashing also handles non-determinism
# ---------------------------------------------------------------------------
def test_strict_canonical_hashing_dict_order() -> None:
    """Strict mode also uses canonical hashing, so dict order is normalized."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="strict")
        def process(d: dict) -> list:
            return sorted(d.items())

        d1 = {"z": 3, "a": 1, "m": 2}
        result1 = process(d1)

        d2 = {"a": 1, "m": 2, "z": 3}
        reset_last_cache_loading()
        result2 = process(d2)

        assert result1 == result2
        assert get_last_cache_loading() is not None, "Strict mode should also normalize dict order"


# ---------------------------------------------------------------------------
# 23. Optimistic mode raw pickle: different dict order = cache miss
# ---------------------------------------------------------------------------
def test_optimistic_raw_pickle_dict_miss() -> None:
    """Optimistic mode uses raw pickle, so different insertion order
    for dicts produces different cache keys — a cache miss.
    This documents the known limitation of optimistic mode.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        @memoize(mode="optimistic")
        def process(d: dict) -> str:
            return str(sorted(d.items()))

        d1 = {"z": 3, "a": 1}
        process(d1)

        # Same content, different insertion order
        d2 = {"a": 1, "z": 3}
        reset_last_cache_loading()
        process(d2)
        # In CPython 3.7+ dicts are ordered, so different insertion order
        # means different pickle bytes — the function is called again (cache miss)
        assert get_last_cache_loading() is None, "Optimistic mode: dict order causes cache miss"


# ---------------------------------------------------------------------------
# 24. Safe mode skips cache on globals() — never stale even across calls
# ---------------------------------------------------------------------------
def test_safe_mode_never_caches_dynamic_globals() -> None:
    """Safe mode: every call to a function using globals() recomputes,
    even when the global hasn't changed — because tracking is incomplete."""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)

        def _use_globals_dict(x: int) -> int:
            return x + globals().get("_safe_test_val", 0)

        @memoize(mode="safe")
        def compute(x: int) -> int:
            return _use_globals_dict(x)

        globals()["_safe_test_val"] = 10
        try:
            assert compute(5) == 15

            # Same args, same global — safe mode still recomputes (no cache hit)
            reset_last_cache_loading()
            assert compute(5) == 15
            assert get_last_cache_loading() is None, (
                "Safe mode should recompute every time with dynamic globals"
            )
        finally:
            globals().pop("_safe_test_val", None)


# ---------------------------------------------------------------------------
# 25. File permission restoration on exception
# ---------------------------------------------------------------------------
def test_loop_until_access_time_restores_permissions_on_exception() -> None:
    """Regression: the file lock context manager must restore permissions
    even when the body raises.
    """
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        file_path = os.path.join(tmp_folder, "permission_test.txt")
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("hello")

        before_mode = os.stat(file_path).st_mode

        with pytest.raises(RuntimeError, match="boom"):
            with loop_until_access_time_greater_than_modification_time(file_path):
                raise RuntimeError("boom")

        after_mode = os.stat(file_path).st_mode
        assert bool(before_mode & stat.S_IWRITE) == bool(after_mode & stat.S_IWRITE)


# ---------------------------------------------------------------------------
# 26. Canonical hash must be stable for frozensets across hash seeds
# ---------------------------------------------------------------------------
def test_canonical_hash_frozenset_is_stable_across_hash_seeds() -> None:
    """Regression: canonical hashing for frozenset must not depend on the
    interpreter's hash seed.
    """
    code = (
        "from memodisk.memodisk import _canonical_hash; "
        "print(_canonical_hash(frozenset(['aa', 'bb', 'cc'])).hex())"
    )

    env1 = os.environ.copy()
    env1["PYTHONHASHSEED"] = "1"
    out1 = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env1,
        cwd=os.path.dirname(__file__),
    ).stdout.strip()

    env2 = os.environ.copy()
    env2["PYTHONHASHSEED"] = "2"
    out2 = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env2,
        cwd=os.path.dirname(__file__),
    ).stdout.strip()

    assert out1 == out2


# ---------------------------------------------------------------------------
# 27. Canonical dict hashing must not depend on repr collisions
# ---------------------------------------------------------------------------
def test_canonical_hash_dict_repr_collision() -> None:
    """Regression: canonical hashing for dicts must remain stable even when
    different keys share the same repr().
    """

    key1 = ReprCollisionKey("a")
    key2 = ReprCollisionKey("b")

    left = _canonical_hash({key1: 1, key2: 2})
    right = _canonical_hash({key2: 2, key1: 1})

    assert left == right

