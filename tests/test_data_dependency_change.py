"""Test with data dependency change."""

import os
import tempfile

from memodisk import (
    get_last_cache_loading,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)


def save_file(filename: str, x: str) -> None:
    print("save_file")
    with open(filename, "w") as fh:
        fh.write(str(x))


@memoize
def load_file_using_context_manager(filename: str) -> str:
    print("load_file")
    with open(filename) as fh:
        line = fh.readline()
    return line


@memoize
def load_file_no_context_manager(filename: str) -> str:
    print("load_file")
    fh = open(filename)
    line = fh.readline()
    fh.close()

    return line


def test_data_dependency_change() -> None:
    """Test that change if data dependency file is detected"""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        print(tmp_folder)
        set_cache_dir(tmp_folder)
        filename = os.path.join(tmp_folder, "test_file.txt")
        save_file(filename, "a")
        time1 = os.stat(filename).st_mtime_ns
        assert load_file_using_context_manager(filename) == "a"

        # test caching works
        assert load_file_using_context_manager(filename) == "a"
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".load_file_using_context_manager")

        # test cache invalidation works
        save_file(filename, "b")
        time2 = os.stat(filename).st_mtime_ns
        print(f"time stamp difference {time2 - time1}")
        assert time2 > time1, f"timestamp difference should strictly positive Got {time2}=={time1}"
        reset_last_cache_loading()
        assert load_file_using_context_manager(filename) == "b"
        assert get_last_cache_loading() is None


def test_data_dependency_change2() -> None:
    """Test that change if data dependency file is detected"""
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        filename = os.path.join(tmp_folder, "test_file.txt")
        save_file(filename, "a")
        time1 = os.stat(filename).st_mtime_ns
        assert load_file_no_context_manager(filename) == "a"

        # test caching works
        assert load_file_no_context_manager(filename) == "a"
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".load_file_no_context_manager")

        # test cache invalidation works
        save_file(filename, "b")
        time2 = os.stat(filename).st_mtime_ns
        print(f"time stamp difference {time2 - time1}")
        assert time2 > time1, f"timestamp difference should strictly positive Got {time2}=={time1}"
        reset_last_cache_loading()
        assert load_file_no_context_manager(filename) == "b"
        assert get_last_cache_loading() is None


if __name__ == "__main__":
    test_data_dependency_change()
    test_data_dependency_change2()
