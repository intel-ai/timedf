# GPu reqs
# !pip -q install ../input/pytorchtabnet/pytorch_tabnet-2.0.1-py3-none-any.whl
import time
from pathlib import Path
import traceback
from contextlib import contextmanager

from utils import TimerManager



# !!!!!!!!!!!!!!!!!!!!!REMOVE ME!!!!!!!!!!!!!
DATA_DIR = Path('/localdisk/benchmark_datasets/optiver_realized_volatility')


def print_trace(name: str = ''):
    print(f'ERROR RAISED IN {name or "anonymous"}')
    print(traceback.format_exc())


def get_workdir_paths(workdir="./optiver_workdir"):
    """Get paths in the workdir, which is shared across several scripts, and create necessary
    folders."""
    workdir = Path(workdir)

    paths = dict(
        workdir=workdir,
        preprocessed=workdir / "features_v2.f",
        artifacts=workdir / "artifacts",
        lfm_features=workdir / "lfm",
        user_features=workdir / "user_features",
    )
    workdir.mkdir(exist_ok=True, parents=True)

    return paths

tm = TimerManager(verbose=True)
timer = tm.timeit
