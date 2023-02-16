# GPu reqs
# !pip -q install ../input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl
import time
from pathlib import Path
import traceback
from contextlib import contextmanager

from utils import TimerManager


def print_trace(name: str = ""):
    print(f'ERROR RAISED IN {name or "anonymous"}')
    print(traceback.format_exc())


def get_workdir_paths(workdir="./optiver_workdir"):
    """Get paths in the workdir, which is shared across several scripts, and create necessary
    folders."""
    workdir = Path(workdir)

    paths = dict(
        workdir=workdir,
        train=workdir / "train.f",
        test=workdir / "test.f",
        folds=workdir / "folds.pkl",
    )
    workdir.mkdir(exist_ok=True, parents=True)

    return paths


tm = TimerManager(verbose=True)
timer = tm.timeit
