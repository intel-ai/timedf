import time
import logging
from typing import Callable, Iterator

__all__ = ["TimerManager"]


logger = logging.getLogger(__name__)


class Timer:
    """Utility timer for TimerManager."""

    def __init__(self, report: Callable[[float], None]) -> None:
        self.report = report
        self.start_time = time.perf_counter()

    def __enter__(self):
        return self

    def stop(self):
        self.report(time.perf_counter() - self.start_time)

    def __exit__(self, type, value, traceback):
        self.stop()


class TimerManager:
    """Utility timer that can measure time using `timeit` function. Intended use is through context manager like
    >>> tm = TimerManager
    >>> with tm.timeit('heavy_call'):
    >>>     heavy_call()
    TimeManager supports nested timings if called through the same object.
    """

    SEPARATOR = "."

    def __init__(self, allow_overwrite=False) -> None:
        self.stack = []
        self.name2time = {}
        self.allow_overwrite = allow_overwrite

    def get_results(self):
        return dict(self.name2time)

    def timeit(self, name):
        self._validate_name(name)
        self._push(name)

        full_name = self._get_full_name()

        def report(time):
            self.report_timer(full_name, time)
            self._pop()

        return Timer(report)

    def report_timer(self, name, time):
        """Record timer result for a timer"""
        if not self.allow_overwrite:
            assert name not in self.name2time, f"Trying to rewrite measurment for {name}"
        logger.info("%s time: %s", name, time)

        self.name2time[name] = time

    def _push(self, name):
        self.stack.append(name)

    def _pop(self):
        self.stack.pop()

    def _validate_name(self, name):
        assert self.SEPARATOR not in name

    def _get_full_name(self):
        return self.SEPARATOR.join(self.stack)
