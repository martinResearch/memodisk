"""Simple memoization example."""

from memodisk import memoize


def fun_a(x: int) -> int:
    print("executing fun_a")
    return x * x * 3


@memoize
def fun_b(x: int) -> int:
    print("executing fun_b")
    return fun_a(x) + 2


if __name__ == "__main__":
    print(f"fun_b(5) = {fun_b(5)}")
