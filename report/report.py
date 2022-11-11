import os
import platform
import re
import socket
import subprocess
from typing import Dict


def enrich_predefined_col2value(col2value: dict) -> dict:
    def get_basic_host_dict():
        return {
            "ServerName": os.environ.get("HOST_NAME", socket.gethostname()),
            "Architecture": platform.architecture()[0],
            "Machine": platform.machine(),
            "Node": platform.node(),
            "OS": platform.system(),
            "CPUCount": os.cpu_count(),
        }

    def match_and_assign(pattern: str, output: str) -> str:
        matches = re.search(pattern, output)
        if matches is None:
            return "N/A"
        if len(matches.groups()) == 1:
            return matches.groups()[0]
        else:
            return "N/A"

    def get_lspcu_dict():
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

    def get_meminfo_dict():
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

    return {
        **get_basic_host_dict(),
        **get_lspcu_dict(),
        **get_meminfo_dict(),
        **col2value,
    }


def get_create_statement(table_name: str, benchmark_specific_col2sql_type: dict, predefined_cols: list) -> str:
    def generate_create_statement(table_name: str, col2sql_spec: dict) -> str:
        return "\n".join(
            [
                f"CREATE TABLE IF NOT EXISTS {table_name} (",
                *[f"    {field} {spec}," for field, spec in col2sql_spec.items()],
                "    PRIMARY KEY (id)" ");",
            ]
        )

    col2sql_spec = {
        "id": "BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT",
        "date": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
        **{name: "VARCHAR(500) NOT NULL" for name in predefined_cols},
        **benchmark_specific_col2sql_type,
    }

    return generate_create_statement(table_name, col2sql_spec)


def get_insert_statement(table_name: str, col2val: dict) -> str:
    def quote_string(string_to_quote: str) -> str:
        if type(n) is str:
            return f"'{string_to_quote}'"
        elif type(string_to_quote) is float and string_to_quote == float("inf"):
            return "4294967295"
        else:
            return str(string_to_quote)

    def generate_insert_statement(table_name: str, col2val: dict) -> str:
        return "\n".join(
            [
                f"INSERT INTO {table_name} (",
                ",".join(col2val),
                ") VALUES(",
                ",".join([quote_string(val) for _, val in col2val.items()]),
                ");",
            ]
        )

    return generate_insert_statement(table_name, col2val)


class DbReport:
    def __init__(
        self,
        database,
        table_name: str,
        benchmark_specific_col2sql_type: Dict[str, str],
        predefined_col2value: Dict[str, str] = {},
    ):
        """Initialize and submit reports to MySQL database

        Parameters
        ----------
        database
            MySQL database from the connector
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
        self._database = database

        self._predefined_col2value = enrich_predefined_col2value(predefined_col2value)
        print("_predefined_field_values = ", self._predefined_col2value)

        statement = get_create_statement(
            table_name=self._table_name,
            benchmark_specific_col2sql_type=benchmark_specific_col2sql_type,
            predefined_cols=list(self._predefined_col2value),
        )
        print("Executing statement", statement)
        self._database.cursor().execute(statement)

    def submit(self, benchmark_col2value: dict):
        statement = get_insert_statement(
            table_name=self._table_name,
            col2val={**self._predefined_col2value, **benchmark_col2value},
        )
        print("Executing statement", statement)
        self._database.cursor().execute(statement)
        self._database.commit()
