"""Testing non pure function."""

from typing import Any, Callable

# from memodisk import get_alias_path
# import inspect

global_list = [45]


def function_using_global_variable_5(x: int) -> int:
    return x * global_list[0]


def call_func(f: Callable, x: Any) -> Any:
    return f(x)


class ClassExample3:
    """Simple class used in test."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: int) -> int:
        return global_list[0] * x


class ClassExample4(ClassExample3):
    """Simple class used in test."""

    def __init__(self) -> None:
        super().__init__()
