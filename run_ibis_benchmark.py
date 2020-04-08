# coding: utf-8
import argparse
import os
import sys
import traceback
import time
import warnings

import mysql.connector

from report import DbReport
from server import OmnisciServer
from server_worker import OmnisciServerWorker
from utils import find_free_port, str_arg_to_bool, check_args_compatibility


def main():
    omniscript_path = os.path.dirname(__file__)
    args = None
    omnisci_server = None
    port_default_value = -1

    benchmarks = ["ny_taxi", "santander", "census", "plasticc"]

    # you can write here a groups of incompatible arguments, for each group to a dict
    # if two of these args in a dict will be equals to its values then '__behavior__' will performs
    # (by default it's 'warning', but also 'terminate' is supported), or you can pass to '__behavior__' a
    # function to execute, or you can pass extra func to '__finally__' that will be execute after 'behavior'
    incompatible_args = [
        {
            "__behavior__": "warning",
            "__finally__": lambda params: params.update({"validation": False}),
            "validation": True,
            "ibis_only": True,
            "no_ibis": True,
        }
    ]

    parser = argparse.ArgumentParser(description="Run internal tests from ibis project")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group("required arguments")
    parser._action_groups.append(optional)

    required.add_argument(
        "-bench_name",
        dest="bench_name",
        choices=benchmarks,
        help="Benchmark name.",
        required=True,
    )
    required.add_argument(
        "-data_file", dest="data_file", help="A datafile that should be loaded.", required=True,
    )
    optional.add_argument(
        "-dfiles_num",
        dest="dfiles_num",
        default=1,
        type=int,
        help="Number of datafiles to input into database for processing.",
    )
    optional.add_argument(
        "-iterations",
        dest="iterations",
        default=1,
        type=int,
        help="Number of iterations to run every query. Best result is selected.",
    )
    optional.add_argument(
        "-dnd", default=False, type=str_arg_to_bool, help="Do not delete old table."
    )
    optional.add_argument(
        "-dni",
        default=False,
        type=str_arg_to_bool,
        help="Do not create new table and import any data from CSV files.",
    )
    optional.add_argument(
        "-validation",
        dest="validation",
        default=False,
        type=str_arg_to_bool,
        help="validate queries results (by comparison with Pandas queries results).",
    )
    optional.add_argument(
        "-import_mode",
        dest="import_mode",
        default="copy-from",
        help="measure 'COPY FROM' import, FSI import, import through pandas",
    )
    optional.add_argument(
        "-optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default="intel",
        help="Which optimizer is used",
    )
    optional.add_argument(
        "-no_ibis",
        default=False,
        type=str_arg_to_bool,
        help="Do not run Ibis benchmark, run only Pandas (or Modin) version",
    )
    optional.add_argument(
        "-ibis_only", default=False, type=str_arg_to_bool, help="Run only Ibis benchmark",
    )
    optional.add_argument(
        "-pandas_mode",
        choices=["Pandas", "Modin_on_ray", "Modin_on_dask", "Modin_on_python"],
        default="Pandas",
        help="Specifies which version of Pandas to use: plain Pandas, Modin runing on Ray or on Dask",
    )
    optional.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. It should have enough space to keep -ray_memory",
    )
    optional.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        help="Size of memory to allocate for Ray plasma store",
    )
    optional.add_argument(
        "-no_ml",
        default=False,
        type=str_arg_to_bool,
        help="Do not run machine learning benchmark, only ETL part",
    )
    optional.add_argument(
        "-gpu_memory",
        dest="gpu_memory",
        type=int,
        help="specify the memory of your gpu, default 16. (This controls the lines to be used. Also work for CPU version. )",
        default=16,
    )
    # MySQL database parameters
    optional.add_argument(
        "-db_server", dest="db_server", default="localhost", help="Host name of MySQL server.",
    )
    optional.add_argument(
        "-db_port", dest="db_port", default=3306, type=int, help="Port number of MySQL server.",
    )
    optional.add_argument(
        "-db_user",
        dest="db_user",
        help="Username to use to connect to MySQL database. "
        "If user name is specified, script attempts to store results in MySQL "
        "database using other -db-* parameters.",
    )
    optional.add_argument(
        "-db_pass",
        dest="db_pass",
        default="omniscidb",
        help="Password to use to connect to MySQL database.",
    )
    optional.add_argument(
        "-db_name",
        dest="db_name",
        default="omniscidb",
        help="MySQL database to use to store benchmark results.",
    )
    optional.add_argument(
        "-db_table_etl",
        dest="db_table_etl",
        help="Table to use to store ETL results for this benchmark.",
    )
    optional.add_argument(
        "-db_table_ml",
        dest="db_table_ml",
        help="Table to use to store ML results for this benchmark.",
    )
    # Omnisci server parameters
    optional.add_argument(
        "-executable", dest="executable", help="Path to omnisci_server executable.",
    )
    optional.add_argument(
        "-omnisci_cwd",
        dest="omnisci_cwd",
        help="Path to omnisci working directory. "
        "By default parent directory of executable location is used. "
        "Data directory is used in this location.",
    )
    optional.add_argument(
        "-port",
        dest="port",
        default=port_default_value,
        type=int,
        help="TCP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-http_port",
        dest="http_port",
        default=port_default_value,
        type=int,
        help="HTTP port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-calcite_port",
        dest="calcite_port",
        default=port_default_value,
        type=int,
        help="Calcite port number to run omnisci_server on.",
    )
    optional.add_argument(
        "-user", dest="user", default="admin", help="User name to use on omniscidb server.",
    )
    optional.add_argument(
        "-password",
        dest="password",
        default="HyperInteractive",
        help="User password to use on omniscidb server.",
    )
    optional.add_argument(
        "-database_name",
        dest="database_name",
        default="omnisci",
        help="Database name to use in omniscidb server.",
    )
    optional.add_argument(
        "-table",
        dest="table",
        default="benchmark_table",
        help="Table name name to use in omniscidb server.",
    )
    optional.add_argument(
        "-ipc_conn",
        dest="ipc_connection",
        default=True,
        type=str_arg_to_bool,
        help="Table name name to use in omniscidb server.",
    )
    # Additional information
    optional.add_argument(
        "-commit_omnisci",
        dest="commit_omnisci",
        default="1234567890123456789012345678901234567890",
        help="Omnisci commit hash to use for benchmark.",
    )
    optional.add_argument(
        "-commit_ibis",
        dest="commit_ibis",
        default="1234567890123456789012345678901234567890",
        help="Ibis commit hash to use for benchmark.",
    )

    try:
        os.environ["PYTHONIOENCODING"] = "UTF-8"
        os.environ["PYTHONUNBUFFERED"] = "1"
        omnisci_server_worker = None
        omnisci_server = None

        args = parser.parse_args()

        if args.port == port_default_value:
            args.port = find_free_port()
        if args.http_port == port_default_value:
            args.http_port = find_free_port()
        if args.calcite_port == port_default_value:
            args.calcite_port = find_free_port()

        if args.bench_name == "ny_taxi":
            from taxi import run_benchmark
        elif args.bench_name == "santander":
            from santander import run_benchmark
        elif args.bench_name == "census":
            from census import run_benchmark
        elif args.bench_name == "plasticc":
            from plasticc import run_benchmark

        parameters = {
            "data_file": args.data_file,
            "dfiles_num": args.dfiles_num,
            "no_ml": args.no_ml,
            "no_ibis": args.no_ibis,
            "optimizer": args.optimizer,
            "pandas_mode": args.pandas_mode,
            "ray_tmpdir": args.ray_tmpdir,
            "ray_memory": args.ray_memory,
            "gpu_memory": args.gpu_memory,
            "validation": args.validation,
            "ibis_only": args.ibis_only,
        }

        if not args.no_ibis:
            if args.executable is None:
                parser.error(
                    "Omnisci executable should be specified with -e/--executable for Ibis part"
                )
            omnisci_server = OmnisciServer(
                omnisci_executable=args.executable,
                omnisci_port=args.port,
                http_port=args.http_port,
                calcite_port=args.calcite_port,
                database_name=args.database_name,
                user=args.user,
                password=args.password,
            )

            parameters["database_name"] = args.database_name
            parameters["table"] = args.table
            parameters["dnd"] = args.dnd
            parameters["dni"] = args.dni
            parameters["import_mode"] = args.import_mode

        check_args_compatibility(parameters, incompatible_args)
        etl_results = []
        ml_results = []
        print(parameters)
        run_id = int(round(time.time()))
        for iter_num in range(1, args.iterations + 1):
            print(f"Iteration #{iter_num}")

            if not args.no_ibis:
                omnisci_server_worker = OmnisciServerWorker(omnisci_server)
                parameters["omnisci_server_worker"] = omnisci_server_worker
                parameters["ipc_connection"] = args.ipc_connection
                omnisci_server.launch()

            result = run_benchmark(parameters)

            if not args.no_ibis:
                omnisci_server_worker.terminate()
                omnisci_server.terminate()

            for backend_res in result["ETL"]:
                if backend_res:
                    backend_res["Iteration"] = iter_num
                    backend_res["run_id"] = run_id
                    etl_results.append(backend_res)
            for backend_res in result["ML"]:
                if backend_res:
                    backend_res["Iteration"] = iter_num
                    backend_res["run_id"] = run_id
                    ml_results.append(backend_res)

            # Reporting to MySQL database
            if args.db_user is not None:
                if iter_num == 1:
                    db = mysql.connector.connect(
                        host=args.db_server,
                        port=args.db_port,
                        user=args.db_user,
                        passwd=args.db_pass,
                        db=args.db_name,
                    )

                    reporting_init_fields = {
                        "OmnisciCommitHash": args.commit_omnisci,
                        "IbisCommitHash": args.commit_ibis,
                    }

                    reporting_fields_benchmark_etl = {
                        x: "VARCHAR(500) NOT NULL" for x in etl_results[0]
                    }
                    if len(etl_results) is not 1:
                        reporting_fields_benchmark_etl.update(
                            {x: "VARCHAR(500) NOT NULL" for x in etl_results[1]}
                        )

                    db_reporter_etl = DbReport(
                        db,
                        args.db_table_etl,
                        reporting_fields_benchmark_etl,
                        reporting_init_fields,
                    )

                    if len(ml_results) is not 0:
                        reporting_fields_benchmark_ml = {
                            x: "VARCHAR(500) NOT NULL" for x in ml_results[0]
                        }
                        if len(ml_results) is not 1:
                            reporting_fields_benchmark_ml.update(
                                {x: "VARCHAR(500) NOT NULL" for x in ml_results[1]}
                            )

                        db_reporter_ml = DbReport(
                            db,
                            args.db_table_ml,
                            reporting_fields_benchmark_ml,
                            reporting_init_fields,
                        )

                for result_etl in etl_results:
                    db_reporter_etl.submit(result_etl)

                if len(ml_results) is not 0:
                    for result_ml in ml_results:
                        db_reporter_ml.submit(result_ml)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
    finally:
        if omnisci_server_worker:
            omnisci_server_worker.terminate()
        if omnisci_server:
            omnisci_server.terminate()


if __name__ == "__main__":
    main()
