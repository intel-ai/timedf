from utils import TimerManager

# Allow overwrite for now, because notebook is currently calling timer many times
# TODO: set to false when accurate benchmarking of each iteration is installed
tm = TimerManager(allow_overwrite=True, verbose=True)
