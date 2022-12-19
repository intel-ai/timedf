import time
import logging


logger = logging.getLogger(__name__)


class TimerManager:
    """
    Utility timer that can measure time using `timeit` function.

    Intended use is through context manager like.

    Notes
    ------
    TimeManager supports nested timings if called through the same object.

    Examples
    ----------
    >>> tm = TimerManager
    >>> with tm.timeit('heavy_call'):
    >>>     heavy_call()
    """

    def __init__(self, allow_overwrite=False) -> None:
        """Initialize root timer.

        Parameters
        ----------
        allow_overwrite, optional
            Allow rewriting of measured time, by default False
        """
        # name for the next timer to start, also acts as timer state
        self.prepared_name = None
        self.timer_stack = self.TimerStack(allow_overwrite=allow_overwrite)

    def timeit(self, name):
        if self.prepared_name is not None:
            raise ValueError(f'Unfinished timer named "{name}" discovered')

        self.prepared_name = name
        return self

    def __enter__(self):
        if self.prepared_name is None:
            raise ValueError("Attempted to start timer, but it has no name")

        self.timer_stack.push(self.prepared_name)
        self.prepared_name = None
        return self

    def __exit__(self, type, value, traceback):
        self.timer_stack.pop()

    def get_results(self):
        return self.timer_stack.get_results()

    class TimerStack:
        """Keeps internal stack of running timers (time and name) and resulting report."""

        SEPARATOR = "."

        def __init__(self, allow_overwrite=False) -> None:
            self.name_stack = []
            self.start_stack = []

            self.allow_overwrite = allow_overwrite
            self.fullname2time = {}

        def push(self, name):
            self._check_name(name)
            self.start_stack.append(time.perf_counter())
            self.name_stack.append(name)

        def pop(self):
            fullname = self._get_full_name()
            self.name_stack.pop()

            self._check_overwrite(fullname)
            self.fullname2time[fullname] = time.perf_counter() - self.start_stack.pop()

        def _check_name(self, name):
            if self.SEPARATOR in name:
                raise ValueError(
                    f'Provided name: "{name}" contains separator symbols "{self.SEPARATOR}"'
                )

        def _check_overwrite(self, fullname):
            if not self.allow_overwrite and fullname in self.fullname2time:
                raise ValueError(f"Trying to rewrite measurment for {fullname}")

        def _get_full_name(self):
            return self.SEPARATOR.join(self.name_stack)

        def get_results(self):
            return dict(self.fullname2time)
