"""Utils to be used by inividual benchmarks"""

import os
import multiprocessing
import time

import psutil

_VM_PEAK_PATTERN = r"VmHWM:\s+(\d+)"


repository_root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
directories = {"repository_root": repository_root_directory}

__all__ = [
    "load_data_pandas",
    "load_data_modin_on_hdk",
    "split",
    "print_results",
    "memory_usage",
    "getsize",
]


def load_data_pandas(
    filename,
    pd,
    columns_names=None,
    columns_types=None,
    header=None,
    nrows=None,
    use_gzip=False,
    parse_dates=None,
):
    types = None
    if columns_types:
        types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    return pd.read_csv(
        filename,
        names=columns_names,
        nrows=nrows,
        header=header,
        dtype=types,
        compression="gzip" if use_gzip else None,
        parse_dates=parse_dates,
    )


def load_data_modin_on_hdk(
    filename, pd, columns_names=None, columns_types=None, parse_dates=None, skiprows=None
):
    dtypes = None
    if columns_types:
        dtypes = {
            columns_names[i]: columns_types[i] if (columns_types[i] != "category") else "string"
            for i in range(len(columns_names))
        }

    all_but_dates = dtypes
    dates_only = False
    if parse_dates:
        parse_dates = parse_dates if isinstance(parse_dates, (list, tuple)) else [parse_dates]
        all_but_dates = {
            col: valtype for (col, valtype) in dtypes.items() if valtype not in parse_dates
        }
        dates_only = [col for (col, valtype) in dtypes.items() if valtype in parse_dates]

    return pd.read_csv(
        filename,
        names=columns_names,
        dtype=all_but_dates,
        parse_dates=dates_only,
        skiprows=skiprows,
    )


def expand_braces(pattern: str):
    """
    Expand braces of the provided string in Linux manner.

    `pattern` should be passed in the next format:
    pattern = "prefix{values_to_expand}suffix"

    Notes
    -----
    `braceexpand` replacement for single string format type.
    Can be used to avoid package import for single corner
    case.

    Examples
    --------
    >>> expand_braces("/taxi/trips_xa{a,b,c}.csv")
    ['/taxi/trips_xaa.csv', '/taxi/trips_xab.csv', '/taxi/trips_xac.csv']
    """
    brace_open_idx = pattern.index("{")
    brace_close_idx = pattern.index("}")

    prefix = pattern[:brace_open_idx]
    suffix = pattern[brace_close_idx + 1 :]
    choices = pattern[brace_open_idx + 1 : brace_close_idx].split(",")

    expanded = []
    for choice in choices:
        expanded.append(prefix + choice + suffix)

    return expanded


def print_results(results, backend=None, ignore_fields=[]):
    if backend:
        print(f"{backend} results:")
    for result_name, result in results.items():
        if result_name not in ignore_fields:
            print("    {} = {:.3f} {}".format(result_name, result, "s"))


# SklearnImport imports sklearn (intel or stock version) only if it is not done previously
class SklearnImport:
    def __init__(self):
        self.current_optimizer = None
        self.train_test_split = None

    def get_train_test_split(self, optimizer):
        assert optimizer is not None, "optimizer parameter should be specified"
        if self.current_optimizer is not optimizer:
            self.current_optimizer = optimizer

            if optimizer == "intel":
                import sklearnex

                sklearnex.patch_sklearn()
                from sklearn.model_selection import train_test_split
            elif optimizer == "stock":
                from sklearn.model_selection import train_test_split
            else:
                raise ValueError(
                    f"Intel optimized and stock sklearn are supported. \
                    {optimizer} can't be recognized"
                )
            self.train_test_split = train_test_split

        return self.train_test_split


sklearn_import = SklearnImport()


def split(X, y, test_size=0.1, stratify=None, random_state=None, optimizer="intel"):
    train_test_split = sklearn_import.get_train_test_split(optimizer)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )

    return X_train, y_train, X_test, y_test


def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)  # GB units


class MemoryTracker:
    __instance = None
    stop_event = multiprocessing.Event()
    data_queue = multiprocessing.Queue()
    tracking_in_progress = False

    @classmethod
    def get_instance(cls):
        """
        Get instance of ``MemoryTracker``.

        Returns
        -------
        MemoryTracker
        """
        if cls.__instance is None:
            cls.__instance = MemoryTracker()
        return cls.__instance

    def start(self):
        """
        Starts tracking memory in a child process.
        """
        self.child = multiprocessing.Process(target=self.track_in_child)
        self.child.start()
        self.tracking_in_progress = True

    def track_in_child(self):
        """
        Function to track memory periodically.

        Tracks memory periodically, when tracking ends peak memory put to data queue.
        """
        max_system_memory = 0
        while not self.stop_event.is_set():
            time.sleep(0.001)
            meminfo_values = self._read_meminfo()
            current_max_system_memory = self._calculate_used_memory(meminfo_values)
            max_system_memory = max(current_max_system_memory, max_system_memory)
            # Htop would show value of (current_max_system_memory / 1024)Gb in Memory bar.
        self.data_queue.put(max_system_memory)

    @staticmethod
    def _read_meminfo():
        """
        Read contents of /proc/meminfo.
        """
        meminfo_values = {}
        with open("/proc/meminfo") as meminfo_file:
            for line in meminfo_file:
                key, value = line.split(":")
                key = key.strip()
                value = int(value.split()[0])  # Extract the numeric value and convert to int
                meminfo_values[key] = value
        return meminfo_values

    @staticmethod
    def _calculate_used_memory(meminfo_values):
        """
        Calculate used memory with the logic used in htop
        https://github.com/htop-dev/htop/blob/main/linux/LinuxMachine.c

        Parameters
        ----------
        meminfo_values : dict
            Content of /proc/meminfo

        Returns
        -------
        float
            Calculated used memory.
        """
        total_mem = meminfo_values.get("MemTotal", 0)
        cached_mem = meminfo_values.get("Cached", 0)
        sreclaimable_mem = meminfo_values.get("SReclaimable", 0)
        free_mem = meminfo_values.get("MemFree", 0)
        buffers_mem = meminfo_values.get("Buffers", 0)
        used_diff = free_mem + cached_mem + sreclaimable_mem + buffers_mem
        used_mem = total_mem - used_diff if total_mem >= used_diff else total_mem - free_mem
        used_mem_mb = used_mem / (1024)
        return used_mem_mb

    def get_memory_used(self):
        """
        Get results of system memory.

        Returns
        -------
        float
            Calculated used memory.
        """
        if self.tracking_in_progress:
            self.stop_event.set()
            self.child.join()
            self.tracking_in_progress = False
            max_system_memory = self.data_queue.get()
        else:
            meminfo_values = self._read_meminfo()
            max_system_memory = self._calculate_used_memory(meminfo_values)
        return max_system_memory


def getsize(filename: str):
    """Return size of filename in MB"""
    if "://" in filename:
        from .s3_client import s3_client

        if s3_client.s3like(filename):
            return s3_client.getsize(filename) / 1024 / 1024
        raise ValueError(f"bad s3like link: {filename}")
    else:
        return os.path.getsize(filename) / 1024 / 1024
