"""Testing memoization with functions using numpy random generator."""

import tempfile

import numpy as np

from memodisk import memoize, set_cache_dir


@memoize
def f1() -> None:
    np.random.rand()
    pass


@memoize
def f2() -> None:
    np.random.seed(2)
    np.random.rand()
    pass


def test_random_state_restoration() -> None:
    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        for _ in range(2):
            np.random.seed(0)
            f1()
            assert np.random.rand() == 0.7151893663724195

            np.random.seed(0)
            f2()
            assert np.random.rand() == 0.025926231827891333


if __name__ == "__main__":
    test_random_state_restoration()
