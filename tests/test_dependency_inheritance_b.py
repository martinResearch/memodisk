"""Testing inherited dependencies from cached subfunction."""

from memodisk import memoize


def f1() -> int:
    return 2


@memoize
def f2() -> int:
    return f1()


@memoize
def f3() -> int:
    return f2()


if __name__ == "__main__":
    assert f3() == 2
