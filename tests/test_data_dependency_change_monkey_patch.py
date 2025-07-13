"""Testing data io functions monkey patching."""

import os
import tempfile

import cv2
import numpy as np

from memodisk import (
    DataLoaderWrapper,
    get_last_cache_loading,
    memoize,
    reset_last_cache_loading,
    set_cache_dir,
)


def save_image(filename: str, x: np.ndarray) -> None:
    cv2.imwrite(filename, x)


@memoize
def load_image(filename: str) -> np.ndarray:
    data = cv2.imread(filename)
    return data  # type: ignore


def test_data_loader_monkey_patching() -> None:
    """Test that change if data dependency file is detected when monkey patching the loading function"""
    cv2.imread = DataLoaderWrapper(
        cv2.imread
    )  # wrap the opencv function for the input file to be added as data dependency

    with tempfile.TemporaryDirectory(prefix="memodisk_cache_tests") as tmp_folder:
        set_cache_dir(tmp_folder)
        reset_last_cache_loading()

        filename = os.path.join(tmp_folder, "test_file.png")
        x = np.zeros((50, 50, 3), dtype=np.uint8)
        save_image(filename, x)
        time1 = os.stat(filename).st_mtime_ns
        assert np.allclose(load_image(filename), x)

        # test caching works
        assert np.allclose(load_image(filename), x)
        last_cache_loading = get_last_cache_loading()
        assert last_cache_loading is not None
        assert last_cache_loading.endswith(".load_image")

        # test cache invalidation works
        x = np.ones((50, 50, 3), dtype=np.uint8)
        save_image(filename, x)
        time2 = os.stat(filename).st_mtime_ns
        print(f"time stamp difference {time2 - time1}")
        assert time2 > time1, f"timestamp difference should strictly positive Got {time2}=={time1}"
        reset_last_cache_loading()
        assert np.allclose(load_image(filename), x)
        assert get_last_cache_loading() is None


if __name__ == "__main__":
    test_data_loader_monkey_patching()
