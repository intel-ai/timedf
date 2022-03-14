import os
import platform
import re
import socket
import subprocess


class DbReport:
    "Initialize and submit reports to MySQL database"

    def __init_predefined_field_values(self, initial_values):
        # System parameters
        self.__predefined_field_values["ServerName"] = (
            os.environ["HOST_NAME"] or socket.gethostname()
        )
        self.__predefined_field_values["Architecture"] = platform.architecture()[0]
        self.__predefined_field_values["Machine"] = platform.machine()
        self.__predefined_field_values["Node"] = platform.node()
        self.__predefined_field_values["OS"] = platform.system()
        self.__predefined_field_values["CPUCount"] = os.cpu_count()

        def match_and_assign(tag, pattern):
            matches = re.search(pattern, output)
            if matches is None:
                return "N/A"
            if len(matches.groups()) == 1:
                return matches.groups()[0]
            else:
                return "N/A"

        # System data from lscpu
        data = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
        output = str(data.communicate()[0].strip().decode())
        lscpu_values = {t: match_and_assign(t, p) for (t, p) in self.__lscpu_patterns.items()}
        self.__predefined_field_values.update(lscpu_values)
        # System data from /proc/meminfo
        with open("/proc/meminfo", "r") as proc_meminfo:
            output = proc_meminfo.read().strip()
            proc_meminfo_values = {
                t: match_and_assign(t, p) for (t, p) in self.__proc_meminfo_patterns.items()
            }
            self.__predefined_field_values.update(proc_meminfo_values)
        # Script specific values
        if initial_values is not None:
            self.__predefined_field_values.update(initial_values)
        print("self.__predefined_field_values = ", self.__predefined_field_values)

    def __init__(self, database, table_name, benchmark_specific_fields, initial_values=None):

        self.__predefined_fields = {
            "id": "BIGINT(20) UNSIGNED NOT NULL AUTO_INCREMENT",
            "ServerName": "VARCHAR(500) NOT NULL",
            "CommitHash": "VARCHAR(500) NOT NULL",
            "date": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
            # System parameters
            "Architecture": "VARCHAR(500) NOT NULL",
            "Machine": "VARCHAR(500) NOT NULL",
            "Node": "VARCHAR(500) NOT NULL",
            "OS": "VARCHAR(500) NOT NULL",
            "CPUCount": "VARCHAR(500) NOT NULL",
            "CPUModel": "VARCHAR(500) NOT NULL",
            "CPUMHz": "VARCHAR(500) NOT NULL",
            "CPUMaxMHz": "VARCHAR(500) NOT NULL",
            "L1dCache": "VARCHAR(500) NOT NULL",
            "L1iCache": "VARCHAR(500) NOT NULL",
            "L2Cache": "VARCHAR(500) NOT NULL",
            "L3Cache": "VARCHAR(500) NOT NULL",
            "MemTotal": "VARCHAR(500) NOT NULL",
            "MemFree": "VARCHAR(500) NOT NULL",
            "MemAvailable": "VARCHAR(500) NOT NULL",
            "SwapTotal": "VARCHAR(500) NOT NULL",
            "SwapFree": "VARCHAR(500) NOT NULL",
            "HugePages_Total": "VARCHAR(500) NOT NULL",
            "HugePages_Free": "VARCHAR(500) NOT NULL",
            "Hugepagesize": "VARCHAR(500) NOT NULL",
        }

        self.__lscpu_patterns = {
            "CPUModel": re.compile("^Model name: +(.+)$", flags=re.MULTILINE),
            "CPUMHz": re.compile("^CPU MHz: +(.+)$", flags=re.MULTILINE),
            "CPUMaxMHz": re.compile("^CPU max MHz: +(.+)$", flags=re.MULTILINE),
            "L1dCache": re.compile("^L1d cache: +(.+)$", flags=re.MULTILINE),
            "L1iCache": re.compile("^L1i cache: +(.+)$", flags=re.MULTILINE),
            "L2Cache": re.compile("^L2 cache: +(.+)$", flags=re.MULTILINE),
            "L3Cache": re.compile("^L3 cache: +(.+)$", flags=re.MULTILINE),
        }
        self.__proc_meminfo_patterns = {
            "MemTotal": re.compile("^MemTotal: +(.+)$", flags=re.MULTILINE),
            "MemFree": re.compile("^MemFree: +(.+)$", flags=re.MULTILINE),
            "MemAvailable": re.compile("^MemAvailable: +(.+)$", flags=re.MULTILINE),
            "SwapTotal": re.compile("^SwapTotal: +(.+)$", flags=re.MULTILINE),
            "SwapFree": re.compile("^SwapFree: +(.+)$", flags=re.MULTILINE),
            "HugePages_Total": re.compile("^HugePages_Total: +(.+)$", flags=re.MULTILINE),
            "HugePages_Free": re.compile("^HugePages_Free: +(.+)$", flags=re.MULTILINE),
            "Hugepagesize": re.compile("^Hugepagesize: +(.+)$", flags=re.MULTILINE),
        }
        self.__predefined_field_values = {}

        self.__table_name = table_name
        self.__init_predefined_field_values(initial_values)
        self.all_fields = self.__predefined_fields
        self.all_fields.update(benchmark_specific_fields)
        self.sql_statement = "CREATE TABLE IF NOT EXISTS %s (" % table_name
        for field, spec in self.all_fields.items():
            self.sql_statement += field + " " + spec + ","
        self.sql_statement += "PRIMARY KEY (id));"
        print("Executing statement", self.sql_statement)
        database.cursor().execute(self.sql_statement)
        self.__database = database

    def __quote_string(self, n):
        if type(n) is str:
            return "'" + n + "'"
        elif type(n) is float:
            if n == float("inf"):
                return "4294967295"
        return str(n)

    def submit(self, benchmark_specific_values):
        self.sql_statement = "INSERT INTO %s (" % self.__table_name
        self.all_fields = self.__predefined_field_values
        self.all_fields.update(benchmark_specific_values)
        for n in list(self.all_fields.keys())[:-1]:
            self.sql_statement += n + ","
        self.sql_statement += list(self.all_fields)[-1] + ") VALUES("
        for n in list(self.all_fields.values())[:-1]:
            self.sql_statement += self.__quote_string(n) + ","
        n = list(self.all_fields.values())[-1]
        self.sql_statement += self.__quote_string(n) + ");"
        print("Executing statement", self.sql_statement)
        self.__database.cursor().execute(self.sql_statement)
        self.__database.commit()
