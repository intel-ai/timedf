import os
import platform
import re
import socket
import subprocess
from typing import Dict, Any, Union, Iterable, Pattern

from sqlalchemy import MetaData, Table, Column, String, DateTime, Integer, func, insert


DEFAULT_STRING_LEN = 500


def enrich_predefined_col2value(col2value: Dict[str, str]) -> Dict[str, str]:
    def get_basic_host_dict() -> Dict[str, Any]:
        return {
            "ServerName": os.environ.get("HOST_NAME", socket.gethostname()),
            "Architecture": platform.architecture()[0],
            "Machine": platform.machine(),
            "Node": platform.node(),
            "OS": platform.system(),
            "CPUCount": os.cpu_count(),
        }

    def match_and_assign(pattern: Union[str, Pattern[str]], output: str) -> str:
        matches = re.search(pattern, output)
        if matches is not None and len(matches.groups()) == 1:
            return matches.groups()[0]
        else:
            return "N/A"

    def get_lspcu_dict() -> Dict[str, str]:
        """System data from lscpu"""

        lscpu_patterns = {
            "CPUModel": re.compile("^Model name: +(.+)$", flags=re.MULTILINE),
            "CPUMHz": re.compile("^CPU MHz: +(.+)$", flags=re.MULTILINE),
            "CPUMaxMHz": re.compile("^CPU max MHz: +(.+)$", flags=re.MULTILINE),
            "L1dCache": re.compile("^L1d cache: +(.+)$", flags=re.MULTILINE),
            "L1iCache": re.compile("^L1i cache: +(.+)$", flags=re.MULTILINE),
            "L2Cache": re.compile("^L2 cache: +(.+)$", flags=re.MULTILINE),
            "L3Cache": re.compile("^L3 cache: +(.+)$", flags=re.MULTILINE),
        }

        data = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
        output = str(data.communicate()[0].strip().decode())
        return {t: match_and_assign(p, output) for t, p in lscpu_patterns.items()}

    def get_meminfo_dict() -> Dict[str, str]:
        """System data from /proc/meminfo"""

        proc_meminfo_patterns = {
            "MemTotal": re.compile("^MemTotal: +(.+)$", flags=re.MULTILINE),
            "MemFree": re.compile("^MemFree: +(.+)$", flags=re.MULTILINE),
            "MemAvailable": re.compile("^MemAvailable: +(.+)$", flags=re.MULTILINE),
            "SwapTotal": re.compile("^SwapTotal: +(.+)$", flags=re.MULTILINE),
            "SwapFree": re.compile("^SwapFree: +(.+)$", flags=re.MULTILINE),
            "HugePages_Total": re.compile("^HugePages_Total: +(.+)$", flags=re.MULTILINE),
            "HugePages_Free": re.compile("^HugePages_Free: +(.+)$", flags=re.MULTILINE),
            "Hugepagesize": re.compile("^Hugepagesize: +(.+)$", flags=re.MULTILINE),
        }

        with open("/proc/meminfo", "r") as proc_meminfo:
            output = proc_meminfo.read().strip()
        return {t: match_and_assign(p, output) for t, p in proc_meminfo_patterns.items()}

    return {**get_basic_host_dict(), **get_lspcu_dict(), **get_meminfo_dict(), **col2value}


def get_table_meta(
    table_name: str,
    str_cols: Iterable[str],
) -> Table:
    return Table(
        table_name,
        MetaData(),
        Column("id", Integer(), primary_key=True),
        Column("date", DateTime(), nullable=False, server_default=func.now()),
        *[Column(name, String(DEFAULT_STRING_LEN), nullable=False) for name in str_cols],
    )


class DbReport:
    def __init__(
        self,
        engine,
        table_name: str,
        benchmark_specific_cols: Iterable[str],
        predefined_col2value: Dict[str, str] = {},
    ):
        """Initialize and submit reports to MySQL database

        Parameters
        ----------
        engine
            Database engine
        table_name
            Table name
        benchmark_specific_col2sql_type
            Declaration of types that will be submitted during benchmarking along with type
            information. For example {'load_data': 'BIGINT UNSIGNED'}.
        predefined_col2value, optional
            Values that are knows before starting the benchmark, they will be submitted along with
            benchmark results, we assume string type for values.
        """
        self._table_name = table_name
        self._engine = engine

        self._predefined_col2value = enrich_predefined_col2value(predefined_col2value)
        print("_predefined_field_values = ", self._predefined_col2value)

        self._table = get_table_meta(
            table_name=self._table_name,
            str_cols=[*self._predefined_col2value, *benchmark_specific_cols],
        )

        self._table.create(self._engine, checkfirst=True)

    def submit(self, benchmark_col2value: Dict[str, Any]):
        with self._engine.connect() as conn:
            conn.execute(
                insert(self._table), [{**self._predefined_col2value, **benchmark_col2value}]
            )
            conn.commit()
