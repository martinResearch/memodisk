"""Test memoized function with change in code."""

import os
import shutil
import subprocess
import sys
import tempfile


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


if __name__ == "__main__":
    test_code_dependency_change()
