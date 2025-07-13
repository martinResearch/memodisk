"""Testing internal functions."""

import inspect

from memodisk import (
    get_function_from_frame,
    get_function_qualified_name_from_frame,
    get_globals_from_code,
)


def test_get_function_qualified_name_from_frame() -> None:
    def f2() -> str:
        frame = inspect.currentframe()
        assert frame is not None
        return get_function_qualified_name_from_frame(frame)

    assert f2() == f2.__qualname__


global_a = 45


def function_using_global_variable(x: int) -> int:
    return x * global_a


def test_get_globals_from_code() -> None:
    global_variables = get_globals_from_code(function_using_global_variable.__code__)
    assert tuple(global_variables) == ("global_a",)


global_a = 50


def test_get_function_from_frame() -> None:
    current_frame = inspect.currentframe()
    assert current_frame is not None
    f = get_function_from_frame(current_frame)
    assert f == test_get_function_from_frame


if __name__ == "__main__":
    inspect.getclosurevars(function_using_global_variable)
    test_get_globals_from_code()
    test_get_function_qualified_name_from_frame()
    test_get_function_from_frame()
