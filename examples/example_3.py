"""Simple memoization example with file io monkey patching."""

import cv2
import numpy as np

from memodisk import DataLoaderWrapper, memoize

# wrap the opencv function for the input file to be added as data dependency

cv2.imread = DataLoaderWrapper(cv2.imread)  # type: ignore


def save_file(x: np.ndarray) -> None:
    cv2.imwrite("test.png", x)


@memoize
def load_file() -> np.ndarray:
    data = cv2.imread("test.png")
    if data is None:
        raise FileNotFoundError("File not found or could not be read.")
    return data


def test() -> None:
    x = np.zeros((50, 50, 3))
    save_file(x)
    assert np.allclose(load_file(), x)
    x = np.ones((50, 50, 3))
    save_file(x)
    assert np.allclose(load_file(), x)


if __name__ == "__main__":
    print("start")
    test()
