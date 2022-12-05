import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    start_time = time.time()
    yield
    print(f"[{name}] {time.time() - start_time:.3f} s")
