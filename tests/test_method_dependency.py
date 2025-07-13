"""Module that contain some failure modes for the library."""

import tempfile

from memodisk import memoize, set_cache_dir


class ClassGlobalVariable:
    """Simple class used in test."""

    def __init__(self) -> None:
        return None

    def fun_a(self) -> str:
        # frame = inspect.currentframe()
        # get_func_name_in_calling_function(frame)
        return "a"

    def fun_b(self) -> str:
        return "b"


global_var = ClassGlobalVariable()
glob_method = global_var.fun_a


@memoize
def function_using_global_method_variable() -> str:
    return glob_method()


def test_function_using_global_method_variable() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        assert function_using_global_method_variable() == "a"
        global glob_method
        glob_method = global_var.fun_b
        assert function_using_global_method_variable() == "b"


if __name__ == "__main__":
    test_function_using_global_method_variable()
