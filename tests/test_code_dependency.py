"""Test memoized function with change in code."""

import datetime
import glob
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pytest

from memodisk import memodisk as memodisk_module
from memodisk import memoize, set_cache_dir


def test_code_dependency_change() -> None:
    """Test that change in a code dependency is detected."""
    folder = os.path.dirname(__file__)
    python_exe = sys.executable

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        print(f"Test done using temp folder {tmp_folder}")
        tmp_file = os.path.join(tmp_folder, "code_test_code_dep_1_tmp.py")

        # first run
        shutil.copyfile(os.path.join(folder, "code_test_code_dep_1_a.py"), tmp_file)
        subprocess.run([python_exe, tmp_file, tmp_folder])

        # second run , check caching works
        result = subprocess.run([python_exe, tmp_file, tmp_folder], stdout=subprocess.PIPE)
        assert b"Result loaded from __main__.fun_b" in result.stdout

        # check invalidation works
        shutil.copyfile(os.path.join(folder, "code_test_code_dep_1_b.py"), tmp_file)
        subprocess.run([python_exe, tmp_file, tmp_folder])


def test_ignore_code_dependency_change_option() -> None:
    """A memoized function can opt out of code-change invalidation."""
    python_exe = sys.executable

    source = textwrap.dedent(
        """
        from memodisk import memoize, set_cache_dir
        import sys

        def helper(x):
            return x * {multiplier}

        @memoize(ignore_code_changes=True)
        def fun_b(x):
            return helper(x) + 2

        if __name__ == "__main__":
            set_cache_dir(sys.argv[1])
            print(fun_b(5))
        """
    )

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        tmp_file = os.path.join(tmp_folder, "ignore_code_dep.py")

        with open(tmp_file, "w", encoding="utf-8") as fh:
            fh.write(source.format(multiplier=3))

        subprocess.run([python_exe, tmp_file, tmp_folder], check=True)

        result = subprocess.run(
            [python_exe, tmp_file, tmp_folder],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert "Result loaded from __main__.fun_b" in result.stdout
        assert result.stdout.strip().endswith("17")

        with open(tmp_file, "w", encoding="utf-8") as fh:
            fh.write(source.format(multiplier=4))

        result = subprocess.run(
            [python_exe, tmp_file, tmp_folder],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert "Result loaded from __main__.fun_b" in result.stdout
        assert result.stdout.strip().endswith("17")


def test_dependency_json_stores_code_file_mtime() -> None:
    @memoize
    def cached_square(x: int) -> int:
        return x * x

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_square(5) == 25

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        assert all_dependencies["code"]
        for entry in all_dependencies["code"]:
            assert "file_last_modified_date_str" in entry
            assert entry["file_last_modified_date_str"] is not None


def test_dependency_check_skips_bytecode_hash_when_code_file_mtime_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @memoize
    def cached_square(x: int) -> int:
        return x * x

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_square(5) == 25

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        def fail_if_rehashed(code):
            raise AssertionError("get_bytecode_hash should not be called when file mtime is unchanged")

        monkeypatch.setattr(memodisk_module, "get_bytecode_hash", fail_if_rehashed)

        assert memodisk_module.dependency_changed(cached_square, all_dependencies) is False


def test_debugpy_line_breakpoint_does_not_change_bytecode_hash() -> None:
    pytest.importorskip("debugpy")
    from debugpy._vendored.pydevd import pydevd
    from debugpy._vendored.pydevd._pydevd_bundle.pydevd_api import PyDevdAPI

    source = textwrap.dedent(
        """
        def target(x):
            y = x + 1
            return y * 2
        """
    )

    with tempfile.TemporaryDirectory(prefix="memodisk_debugpy_tests") as tmp_folder:
        module_path = os.path.join(tmp_folder, "breakpoint_target.py")
        with open(module_path, "w", encoding="utf-8") as fh:
            fh.write(source)

        spec = importlib.util.spec_from_file_location("breakpoint_target", module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        target = module.target
        filename = os.path.normpath(target.__code__.co_filename)
        line = target.__code__.co_firstlineno + 1

        before_code = target.__code__.co_code
        before_hash = memodisk_module.get_bytecode_hash(target.__code__)

        py_db = pydevd.PyDB(set_as_global=False)
        api = PyDevdAPI()

        try:
            api.run(py_db)
            api.set_ide_os_and_breakpoints_by(py_db, 0, "WINDOWS", "LINE")
            result = api.add_breakpoint(
                py_db,
                filename,
                "python-line",
                1,
                line,
                None,
                "None",
                None,
                "NONE",
                None,
                False,
                adjust_line=True,
            )

            assert result.error_code == 0
            assert target(10) == 22
            assert target.__code__.co_code == before_code
            assert memodisk_module.get_bytecode_hash(target.__code__) == before_hash
        finally:
            api.remove_all_breakpoints(py_db, "*")
            py_db.disable_tracing()


def test_dependency_json_stores_site_packages_package_versions() -> None:
    np = pytest.importorskip("numpy")

    @memoize
    def cached_sin(x: float) -> float:
        return float(np.sin(x))

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_sin(1.0) == pytest.approx(float(np.sin(1.0)))

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        package_versions = {
            package_name: version
            for entry in all_dependencies["code"]
            for package_name, version in entry.get("package_versions", {}).items()
        }
        assert package_versions["numpy"] == importlib.metadata.version("numpy")


def test_dependency_changed_detects_site_packages_package_version_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")

    @memoize
    def cached_sin(x: float) -> float:
        return float(np.sin(x))

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_sin(1.0) == pytest.approx(float(np.sin(1.0)))

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        real_version = importlib.metadata.version("numpy")

        def fake_package_version(package_name: str) -> str | None:
            if package_name == "numpy":
                return "0.0.0-test"
            return real_version if package_name == "numpy" else None

        monkeypatch.setattr(memodisk_module, "_get_installed_package_version", fake_package_version)

        assert memodisk_module.dependency_changed(cached_sin, all_dependencies) is True


def test_dependency_json_stores_compiled_module_dependencies() -> None:
    np = pytest.importorskip("numpy")

    @memoize
    def cached_sin(x: float) -> float:
        return float(np.sin(x))

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_sin(1.0) == pytest.approx(float(np.sin(1.0)))

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        compiled_dependencies = {
            file_path: last_modified_date_str
            for entry in all_dependencies["code"]
            for file_path, last_modified_date_str in entry.get("compiled_dependencies", {}).items()
        }
        assert compiled_dependencies
        assert any(file_path.endswith(".pyd") for file_path in compiled_dependencies)


def test_dependency_changed_detects_compiled_module_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")

    @memoize
    def cached_sin(x: float) -> float:
        return float(np.sin(x))

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_sin(1.0) == pytest.approx(float(np.sin(1.0)))

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        compiled_dependencies = {
            file_path: last_modified_date_str
            for entry in all_dependencies["code"]
            for file_path, last_modified_date_str in entry.get("compiled_dependencies", {}).items()
        }
        target_file = next(iter(compiled_dependencies))

        real_get_file_last_modified_date_str = memodisk_module._get_file_last_modified_date_str

        def fake_get_file_last_modified_date_str(file_path: str) -> str | None:
            if file_path == target_file:
                return "1900-01-01 00:00:00"
            return real_get_file_last_modified_date_str(file_path)

        monkeypatch.setattr(memodisk_module, "_get_file_last_modified_date_str", fake_get_file_last_modified_date_str)

        assert memodisk_module.dependency_changed(cached_sin, all_dependencies) is True


def test_dependency_json_stores_ambient_time_sources() -> None:
    dt = datetime.datetime

    @memoize(mode="optimistic")
    def cached_now() -> str:
        return dt.now().isoformat()

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_now()

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        assert all_dependencies["ambient_time_sources"] == ["datetime.datetime.now"]


def test_dependency_json_stores_external_executable_dependency() -> None:
    @memoize(mode="optimistic")
    def cached_subprocess_version() -> str:
        return subprocess.check_output(
            [sys.executable, "-c", "print('ok')"],
            text=True,
        ).strip()

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_subprocess_version() == "ok"

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        data_dependencies = {entry["file_path"] for entry in all_dependencies["data"]}
        assert os.path.abspath(sys.executable) in data_dependencies
        assert all_dependencies["external_processes"] == [
            {
                "executable_path": os.path.abspath(sys.executable),
                "tracking_kind": "direct",
                "tracking_complete": False,
            }
        ]


def test_dependency_changed_detects_external_executable_change() -> None:
    @memoize(mode="optimistic")
    def cached_subprocess_version() -> str:
        return subprocess.check_output(
            [sys.executable, "-c", "print('ok')"],
            text=True,
        ).strip()

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert cached_subprocess_version() == "ok"

        dep_files = glob.glob(os.path.join(tmp_folder, "*_dependencies.json"))
        assert len(dep_files) == 1

        with open(dep_files[0], encoding="utf-8") as fh:
            all_dependencies = json.load(fh)

        executable_path = os.path.abspath(sys.executable)
        for entry in all_dependencies["data"]:
            if os.path.abspath(entry["file_path"]) == executable_path:
                entry["last_modified_date_str"] = "1900-01-01 00:00:00"
                break
        else:
            raise AssertionError("Expected sys.executable to be tracked as a data dependency")

        assert memodisk_module.dependency_changed(cached_subprocess_version, all_dependencies) is True


if __name__ == "__main__":
    test_code_dependency_change()
