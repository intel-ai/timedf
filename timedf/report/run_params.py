"""Module with classes that encapsulate generation and extraction of benchmark run parameters."""

import os
import platform
import re
import socket
import subprocess
from timedf.benchmark_utils import MemoryTracker
from typing import Dict, Any, Union, Pattern
import warnings


def _get_host_info() -> Dict[str, str]:
    def get_basic_host_dict() -> Dict[str, Any]:
        return {
            "server_name": os.environ.get("HOST_NAME", socket.gethostname()),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "node": platform.node(),
            "os": platform.system(),
            "cpu_count": os.cpu_count(),
        }

    def match_and_assign(pattern: Union[str, Pattern[str]], output: str) -> str:
        matches = re.search(pattern, output)
        if matches is not None and len(matches.groups()) == 1:
            return matches.groups()[0]
        else:
            return ""

    def get_lspcu_dict() -> Dict[str, str]:
        """System data from lscpu"""

        lscpu_patterns = {
            "cpu_model": re.compile("^Model name: +(.+)$", flags=re.MULTILINE),
            "cpu_mhz": re.compile("^CPU MHz: +(.+)$", flags=re.MULTILINE),
            "cpu_max_mhz": re.compile("^CPU max MHz: +(.+)$", flags=re.MULTILINE),
            "cpu_l1d_cache": re.compile("^L1d cache: +(.+)$", flags=re.MULTILINE),
            "cpu_l1i_cache": re.compile("^L1i cache: +(.+)$", flags=re.MULTILINE),
            "cpu_l2_cache": re.compile("^L2 cache: +(.+)$", flags=re.MULTILINE),
            "cpu_l3_cache": re.compile("^L3 cache: +(.+)$", flags=re.MULTILINE),
        }

        try:
            data = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
            output = str(data.communicate()[0].strip().decode())
        except FileNotFoundError:
            warnings.warn(
                "Couldn't run `lscpu` is this linux? Description of host machine will be"
                " incomplete"
            )
            output = ""
        return {t: match_and_assign(p, output) for t, p in lscpu_patterns.items()}

    def get_meminfo_dict() -> Dict[str, str]:
        """System data from /proc/meminfo"""

        proc_meminfo_patterns = {
            "mem_total": re.compile("^MemTotal: +(.+)$", flags=re.MULTILINE),
            "mem_free": re.compile("^MemFree: +(.+)$", flags=re.MULTILINE),
            "mem_available": re.compile("^MemAvailable: +(.+)$", flags=re.MULTILINE),
            "swap_total": re.compile("^SwapTotal: +(.+)$", flags=re.MULTILINE),
            "swap_free": re.compile("^SwapFree: +(.+)$", flags=re.MULTILINE),
            "huge_pages_total": re.compile("^HugePages_Total: +(.+)$", flags=re.MULTILINE),
            "huge_pages_free": re.compile("^HugePages_Free: +(.+)$", flags=re.MULTILINE),
            "hugepage_size": re.compile("^Hugepagesize: +(.+)$", flags=re.MULTILINE),
        }

        try:
            with open("/proc/meminfo", "r") as proc_meminfo:
                output = proc_meminfo.read().strip()
        except FileNotFoundError:
            warnings.warn(
                "Couldn't open `/proc/meminfo` is this linux?\n"
                "Description of host machine will be incomplete"
            )
            output = ""

        return {t: match_and_assign(p, output) for t, p in proc_meminfo_patterns.items()}

    max_system_memory = MemoryTracker.get_instance().get_memory_used()
    print(f"max_system_memory = {max_system_memory}")
    if max_system_memory is not None:
        max_memory_mb = str(int(max_system_memory))

    return {
        **get_basic_host_dict(),
        **get_lspcu_dict(),
        **get_meminfo_dict(),
        "max_memory_mb": max_memory_mb,
    }


class HostParams:
    fields = tuple(_get_host_info())

    def prepare_report_dict(self):
        return _get_host_info()


class RunParams:
    fields = (
        "data_file",
        "no_ml",
        "use_modin_xgb",
        "ray_tmpdir",
        "ray_memory",
        "num_threads",
        # Optional tag to label specific runs by user
        "tag",
    )

    def _validate_params(self, params):
        diff = set(self.fields) - set(params)
        if len(diff) > 0:
            raise ValueError(f"The following params are missing: {diff}")

    def prepare_report_dict(self, params):
        self._validate_params(params)
        return {name: params[name] for name in self.fields}
