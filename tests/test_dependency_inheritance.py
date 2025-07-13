"""Test with dependency inheritance when nested function are memoized."""

import os
import shutil
import subprocess
import sys
import tempfile


def test_dependency_inheritance() -> None:
    """Test that change in a code dependency is detected"""
    folder = os.path.dirname(__file__)
    python_exe = sys.executable

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        print(f"Test done using temp folder {tmp_folder}")
        tmp_file = os.path.join(tmp_folder, "test3.py")

        # first run
        shutil.copyfile(os.path.join(folder, "test_dependency_inheritance_a.py"), tmp_file)
        result = subprocess.run([python_exe, tmp_file, tmp_folder])

        # second run , check caching works
        result = subprocess.run([python_exe, tmp_file, tmp_folder], stdout=subprocess.PIPE)
        print(result.stdout)
        assert b"Result loaded from __main__.f3" in result.stdout

        # check invalidation works
        shutil.copyfile(os.path.join(folder, "test_dependency_inheritance_b.py"), tmp_file)
        subprocess.run([python_exe, tmp_file, tmp_folder])


if __name__ == "__main__":
    test_dependency_inheritance()
