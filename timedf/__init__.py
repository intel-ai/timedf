from .benchmark import BaseBenchmark, BenchmarkResults
from .timer import TimerManager, tm
from . import benchmark_utils

# ! Do not store pandas_backend here, it will potentially store just pandas, without chaning it on
# command

from .__version__ import __version__