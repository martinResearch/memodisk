"""Testing inherited dependencies from cached subfunction."""

from memodisk import memoize


def f1() -> int:
    return 1


@memoize
def f2() -> int:
    return f1()


@memoize
def f3() -> int:
    return f2()


if __name__ == "__main__":
    # call f2 to memoize it. Next time f1 won't be called
    # and won't appear in dependencies of f3 is memoized result of f2 is used
    f2()
    assert f3() == 1
