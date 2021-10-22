import json
import itertools as it
import math
import argparse
from utils_base_env import add_mysql_arguments


class DbReport:
    """Initialize and submit reports to MySQL database"""

    def __init__(self, db_ops, table_name, persistent_values=None):
        import mysql.connector

        db = mysql.connector.connect(**db_ops)
        self._database = db

        self.all_fields = {}
        self._table_name = table_name

        if persistent_values:
            self.all_fields.update(persistent_values)

    def __quote_string(self, n):
        if type(n) is str:
            return "'" + n + "'"
        elif type(n) is float:
            if n == float("inf"):
                return "4294967295"
        return str(n)

    def submit(self, submit_values):
        def submit_row(row_values):
            self.sql_statement = f"INSERT INTO {self._table_name} ("
            self.all_fields.update(row_values)

            for n in list(self.all_fields.keys())[:-1]:
                self.sql_statement += n + ","
            self.sql_statement += list(self.all_fields)[-1] + ") VALUES("
            for n in list(self.all_fields.values())[:-1]:
                self.sql_statement += self.__quote_string(n) + ","
            n = list(self.all_fields.values())[-1]
            self.sql_statement += self.__quote_string(n) + ");"
            self.sql_statement = self.sql_statement.replace("'NULL'", "NULL")
            self._database.cursor().execute(self.sql_statement)

        if isinstance(submit_values, (list, set)):
            for row_values in submit_values:
                submit_row(row_values)
        elif isinstance(submit_values, dict):
            submit_row(submit_values)
        else:
            raise TypeError(f"Unsupported `submit_values` type: {type(submit_values)}")

        self._database.commit()


def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-path",
        dest="result_path",
        type=str,
        required=True,
        default=None,
        help="File path of ASV result to report.",
    )
    add_mysql_arguments(parser)
    parsed_args = parser.parse_args()
    db_ops = {}
    for arg in (
        ("host", "db_server"),
        ("port", "db_port"),
        ("user", "db_user"),
        ("passwd", "db_pass"),
        ("db", "db_name"),
    ):
        db_ops[arg[0]] = getattr(parsed_args, arg[1])
    return parsed_args.result_path, db_ops


def parse_asv_results(result_path):
    results = []

    with open(result_path, "r") as f:
        res = json.load(f)

    reporting_init_fields = {
        "ServerName": res["params"]["machine"],
        "Machine": res["params"]["arch"],
        "CPUModel": res["params"]["cpu"],
        "CPUCount": res["params"]["num_cpu"],
        "OS": res["params"]["os"],
        "RAM": res["params"]["ram"],
        "ModinCommitHash": res["commit_hash"],
    }

    for benchmark in res["results"]:
        counter = 0
        bench_result = res["results"][benchmark]
        nested_lists = isinstance(bench_result, dict)
        params = (
            res["results"][benchmark]["params"] if nested_lists else res["results"][benchmark][1]
        )
        combinations = list(it.product(*params))
        # parameters `combinations` represented as strings for each of benchmark runs
        combinations_str = ["_".join([str(param) for param in comb]) for comb in combinations]
        result_field = bench_result["result"] if nested_lists else bench_result[0]
        for comb in combinations_str:
            comb_result = (
                result_field[counter]
                if result_field is not None and result_field[counter] is not None
                else "NULL"
            )
            comb_result = (
                "NULL"
                if isinstance(comb_result, float) and math.isnan(comb_result)
                else comb_result
            )
            results.append(
                {
                    "Benchmark": benchmark,
                    "Parameters": comb.replace("'", ""),
                    "Result": comb_result,
                }
            )
            counter += 1

    return reporting_init_fields, results


def main():
    result_path, db_ops = get_cmd_args()
    reporting_init_fields, results = parse_asv_results(result_path)
    db_table = "25_modin_funcs"

    db_reporter = DbReport(db_ops, db_table, reporting_init_fields)
    db_reporter.submit(results)
    print("Data was successfully reported!")


if __name__ == "__main__":
    main()
