"""Testing internal functions."""

import inspect

from memodisk import (
    get_globals_from_code,
)

global_a = 45


def function_using_global_variable(x: int) -> int:
    return x * global_a


def test_get_globals_from_code() -> None:
    global_variables = get_globals_from_code(function_using_global_variable.__code__)
    assert tuple(global_variables) == ("global_a",)


global_a = 50


if __name__ == "__main__":
    inspect.getclosurevars(function_using_global_variable)
    test_get_globals_from_code()
