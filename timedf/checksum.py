import time


class ChecksumStore:
    """Store to record checksums.

    Example use:

    >>> cs = ChecksumStore()

    >>> with cs.record_checksum("split_size") as c:
    >>>    c.set(df["t"].sum())

    """

    def __init__(self) -> None:
        self.store = {}

    def check(self, name):
        assert name not in self.store

        return self.Checksum(name, self)

    def _record(self, name, value, duration_s):
        assert name not in self.store

        self.store[name] = {"duration_s": duration_s, "value": value}

    def get_results(self):
        return dict(self.store)

    class Checksum:
        def __init__(self, name, store) -> None:
            self.name = name
            self.store = store
            self.value = None
            self.start = None

        def set(self, value):
            self.value = value

        def __enter__(self):
            assert self.start is None
            assert self.value is None

            self.start = time.perf_counter()
            return self

        def __exit__(self, type, value, traceback):
            assert self.start is not None
            assert self.value is not None

            duration_s = time.perf_counter() - self.start
            self.store._record(name=self.name, duration_s=duration_s, value=self.value)


cs = ChecksumStore()
