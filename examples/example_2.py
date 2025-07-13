"""Simple memoization example with file io."""

from memodisk import memoize


def save_file(x: str) -> None:
    print("save_file")
    fh = open("test_file.txt", "w")
    fh.write(str(x))


@memoize
def load_file() -> str:
    print("load_file")
    fh = open("test_file.txt")
    line = fh.readline()
    return line


def test() -> None:
    save_file("a")
    assert load_file() == "a"
    save_file("b")
    assert load_file() == "b"


if __name__ == "__main__":
    test()
