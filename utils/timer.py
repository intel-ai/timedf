import time
from typing import Callable

__all__ = ["TimerManager"]


class Timer:
    def __init__(self, report: Callable[[float], None]) -> None:
        self.report = report
        self.start_time = time.perf_counter()

    def __enter__(self):
        return self

    def stop_timer(self):
        self.report(time.perf_counter() - self.start_time)

    def __exit__(self, type, value, traceback):
        self.stop_timer()


class TimerManager:
    SEPARATOR = "."

    def __init__(self, allow_overwrite=False) -> None:
        self.stack = []
        self.name2time = {}
        self.allow_overwrite = allow_overwrite

    def report_timer(self, name, time):
        if not self.allow_overwrite:
            assert name not in self.name2time, f"Trying to rewrite measurment for {name}"
        self.name2time[name] = time

    def timeit(self, name):
        assert self.SEPARATOR not in name

        self.stack.append(name)

        name = self.SEPARATOR.join(self.stack)

        def report(time):
            self.report_timer(name, time)
            self.stack.pop()

        return Timer(report)

    def get_results(self):
        return dict(self.name2time)
