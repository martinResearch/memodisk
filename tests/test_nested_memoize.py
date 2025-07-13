"""Testing nested functions memoization"""

from memodisk import memoize


@memoize
def fun_a(x: int) -> int:
    print("executing fun_a")
    return x * x * 3


@memoize
def fun_b(x: int) -> int:
    return fun_a(x) + 2


def test_nested() -> None:
    fun_b(5)


if __name__ == "__main__":
    test_nested()
