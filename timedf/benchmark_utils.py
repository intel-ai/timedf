"""Utils to be used by inividual benchmarks"""
import os
import re
import multiprocessing
import platform
import time
import warnings

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
        self.child = multiprocessing.Process(target=self.track_in_child)
        self.child.start()
        self.tracking_in_progress = True
        
    def track_in_child(self):
        max_memory_system = 0
        while not self.stop_event.is_set():
            time.sleep(1)            
            meminfo_values = self._read_meminfo()
            current_max_memory_system = self._calculate_used_memory(meminfo_values)
            max_memory_system=max(current_max_memory_system,max_memory_system)
            print(current_max_memory_system)
        self.data_queue.put(max_memory_system)  
    
    @staticmethod
    def _read_meminfo():
        meminfo_values = {}
        with open('/proc/meminfo') as meminfo_file:
            for line in meminfo_file:
                key, value = line.split(':')
                key = key.strip()
                value = int(value.split()[0])  # Extract the numeric value and convert to int
                meminfo_values[key] = value
        return meminfo_values
    
    @staticmethod
    def _calculate_used_memory(meminfo_values):
        total_mem = meminfo_values.get('MemTotal', 0)
        cached_mem = meminfo_values.get('Cached', 0)
        sreclaimable_mem = meminfo_values.get('SReclaimable', 0)
        free_mem = meminfo_values.get('MemFree', 0)
        buffers_mem = meminfo_values.get('Buffers', 0)
        used_diff = free_mem + cached_mem + sreclaimable_mem + buffers_mem
        usedMem = total_mem - used_diff if total_mem >= used_diff else total_mem - free_mem
        used_mem_gb = usedMem/1024**2
        return used_mem_gb

    def _calculate_pss_total(self,pid):
        try:
            # Open the smaps file and read line by line
            with open(f'/proc/{pid}/smaps', 'r') as smaps_file:
                total_pss = 0

                for line in smaps_file:
                    # Check if the line starts with 'Pss'
                    if line.startswith('Pss'):
                        # Split the line and add the second field (index 1) to the total
                        total_pss += float(line.split()[1])

                return total_pss

        except FileNotFoundError:
            print("/proc/smaps not found. Make sure you're running on a Linux-like system.")
            return None

        except Exception as e:
            print(f"Error: {e}")
            return None

    
    def get_result(self):
        if self.tracking_in_progress:
            self.stop_event.set()
            self.child.join()
            self.tracking_in_progress = False
            max_memory_rss,max_memory_system = self.data_queue.get()
            print(f"finaly max_memory_system ={max_memory_system}, max_memory_rss={max_memory_rss} ")
        else:
            max_memory_rss = sum([self._calculate_pss_total(process.pid)  for process in LaunchedProcesses.get_instance().process_list])
            max_memory_system = psutil.virtual_memory().total - psutil.virtual_memory().free
            
         
        return max_memory_system, max_memory_rss 

class LaunchedProcesses:
    """
    Keep track of processes launched for running the timedf benchmark.
    The process list would contain a single process for all backends
    except for `Modin_on_unidist_mpi`, which would contain multiple processes
    if unidist on MPI is launched in SPMD mode.
    """

    __instance = None
    process_list = [psutil.Process()]
    
    

    @classmethod
    def get_instance(cls):
        """
        Get instance of ``LaunchedProcesses``.

        Returns
        -------
        LaunchedProcesses
        """
        if cls.__instance is None:
            cls.__instance = LaunchedProcesses()
        return cls.__instance

    def set_process_list(self, process_list):
        self.process_list = process_list

    def get_process_list(self):
        return self.process_list


class LaunchedProcesses:
    """
    Keep track of processes launched for running the timedf benchmark.
    The process list would contain a single process for all backends
    except for `Modin_on_unidist_mpi`, which would contain multiple processes
    if unidist on MPI is launched in SPMD mode.
    """

    __instance = None
    process_list = [psutil.Process()]

    @classmethod
    def get_instance(cls):
        """
        Get instance of ``LaunchedProcesses``.

        Returns
        -------
        LaunchedProcesses
        """
        if cls.__instance is None:
            cls.__instance = LaunchedProcesses()
        return cls.__instance

    def set_process_list(self, process_list):
        self.process_list = process_list

    def get_process_list(self):
        return self.process_list


def get_max_memory_usage(proc=psutil.Process()):
    """Reads maximum memory usage in MB from process history. Returns 0 on non-linux systems
    or if the process is not alive."""
    max_mem = 0
    try:
        with open(f"/proc/{proc.pid}/status", "r") as stat:
            for match in re.finditer(_VM_PEAK_PATTERN, stat.read()):
                max_mem = float(match.group(1))
                # MB conversion
                max_mem = int(max_mem / 1024)
                break
    except FileNotFoundError:
        if platform.system() == "Linux":
            warnings.warn(f"Couldn't open `/proc/{proc.pid}/status` file. Is the process alive?")
        else:
            warnings.warn("Couldn't get the max memory usage on a non-Linux platform.")
        return 0

    return max_mem + sum(get_max_memory_usage(c) for c in proc.children())

def get_pss_usage(proc):
    """Reads maximum memory usage in MB from process history. Returns 0 on non-linux systems
    or if the process is not alive."""
    max_mem = 0
    try:
            # Open the smaps file and read line by line
        with open(f'/proc/{proc.pid}/smaps', 'r') as smaps_file:
            total_pss = 0

            for line in smaps_file:
                # Check if the line starts with 'Pss'
                if line.startswith('Pss'):
                    # Split the line and add the second field (index 1) to the total
                    total_pss += float(line.split()[1])

            return total_pss

    except FileNotFoundError:
        print("/proc/smaps not found. Make sure you're running on a Linux-like system.")
        return None

    except Exception as e:
        print(f"Error: {e}")
        return None

    return max_mem + sum(get_pss_usage(c) for c in proc.children())


def getsize(filename: str):
    """Return size of filename in MB"""
    if "://" in filename:
        from .s3_client import s3_client

        if s3_client.s3like(filename):
            return s3_client.getsize(filename) / 1024 / 1024
        raise ValueError(f"bad s3like link: {filename}")
    else:
        return os.path.getsize(filename) / 1024 / 1024
