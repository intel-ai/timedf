from braceexpand import braceexpand
import mysql.connector
import argparse
import pathlib
import glob
import json
import copy
import sys
import os
from server import OmnisciServer
from report import DbReport
from environment import CondaEnvironment
from utils import str_arg_to_bool

omniscript_path = os.path.dirname(__file__)
omnisci_server = None
args = None


def execute_benchmark(
    datafiles,
    import_cmdline,
    benchmark_cwd,
    benchmark_cmdline,
    fragment_size,
    results_file_name,
    report,
    conda_env,
):
    if import_cmdline is not None:
        ic = copy.copy(import_cmdline)
        # Import dataset mode
        if fragment_size is not None:
            ic += ["--fragment-size", str(fragment_size)]
            fs = fragment_size
        else:
            fs = 0
        print("IMPORT COMMAND LINE:", ic)
        conda_env.run(ic)
    else:
        # Synthetic benchmark mode
        benchmark_cmdline += ["--fragment_size", str(fragment_size)]
        fs = fragment_size

    # Execute benchmark
    print("BENCHMARK COMMAND LINE", benchmark_cmdline)
    conda_env.run(benchmark_cmdline, cwd=benchmark_cwd)

    # Parse report
    with open(results_file_name, "r") as results_file:
        results = json.load(results_file)
    for result in results:
        print(
            datafiles,
            ",",
            fs,
            ",",
            result["name"],
            ",",
            result["results"]["query_exec_min"],
            ",",
            result["results"]["query_total_min"],
            ",",
            result["results"]["query_exec_max"],
            ",",
            result["results"]["query_total_max"],
            ",",
            result["results"]["query_exec_avg"],
            ",",
            result["results"]["query_total_avg"],
            ",",
            result["results"]["query_error_info"],
            "\n",
            file=report,
            sep="",
            end="",
            flush=True,
        )
        if db_reporter is not None:
            db_reporter.submit(
                {
                    "FilesNumber": datafiles,
                    "FragmentSize": fs,
                    "BenchName": result["name"],
                    "BestExecTimeMS": str(result["results"]["query_exec_min"]),
                    "BestTotalTimeMS": result["results"]["query_total_min"],
                    "WorstExecTimeMS": str(result["results"]["query_exec_max"]),
                    "WorstTotalTimeMS": result["results"]["query_total_max"],
                    "AverageExecTimeMS": str(result["results"]["query_exec_avg"]),
                    "AverageTotalTimeMS": result["results"]["query_total_avg"],
                }
            )


parser = argparse.ArgumentParser(
<<<<<<< HEAD
    description="Run arbitrary omnisci benchmark and submit report "
    "values to MySQL database"
=======
    description="Run arbitrary omnisci benchmark and submit report " "values to MySQL database"
>>>>>>> go/master
)
optional = parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
parser._action_groups.append(optional)

# Environment
required.add_argument(
    "-en", "--env_name", dest="env_name", default="ibis-tests", help="Conda env name."
)
optional.add_argument(
    "-ec",
    "--env_check",
    dest="env_check",
    default=False,
    type=str_arg_to_bool,
    help="Check if env exists. If it exists don't recreate.",
)
optional.add_argument(
    "-s",
    "--save_env",
    dest="save_env",
    default=False,
    type=str_arg_to_bool,
    help="Save conda env after executing.",
)
optional.add_argument(
    "-ci",
    "--ci_requirements",
    dest="ci_requirements",
    default=None,
    help="File with ci requirements for conda env.",
)

# Benchmark scripts location
required.add_argument(
    "-r", "--report", dest="report", default="report.csv", help="Report file name"
)
required.add_argument(
<<<<<<< HEAD
    "-path",
    dest="benchmarks_path",
    required=True,
    help="Path to omniscidb/Benchmarks directory.",
=======
    "-path", dest="benchmarks_path", required=True, help="Path to omniscidb/Benchmarks directory."
>>>>>>> go/master
)
# Omnisci server parameters
required.add_argument(
    "-e",
    "--executable",
    dest="omnisci_executable",
    required=True,
    help="Path to omnisci_server executable.",
)
optional.add_argument(
    "-w",
    "--workdir",
    dest="omnisci_cwd",
    help="Path to omnisci working directory. By default parent directory of "
    "executable location is used. Data directory is used in this location.",
)
optional.add_argument(
    "-o",
    "--port",
    dest="omnisci_port",
    default=62274,
    type=int,
    help="TCP port number to run omnisci_server on.",
)
required.add_argument(
    "-u",
    "--user",
    dest="user",
    default="admin",
    required=True,
    help="User name to use on omniscidb server.",
)
required.add_argument(
    "-p",
    "--passwd",
    dest="passwd",
    default="HyperInteractive",
    required=True,
    help="User password to use on omniscidb server.",
)
required.add_argument(
    "-n",
    "--name",
    dest="name",
    default="omnisci",
    required=True,
    help="Database name to use on omniscidb server.",
)
required.add_argument(
    "-t",
    "--import-table-name",
    dest="import_table_name",
    required=True,
    help="Name of table to import data to. NOTE: This table will be dropped "
    "before and after the import test.",
)
# Required by omnisci benchmark scripts
<<<<<<< HEAD
required.add_argument(
    "-l", "--label", dest="label", required=True, help="Benchmark run label."
)
=======
required.add_argument("-l", "--label", dest="label", required=True, help="Benchmark run label.")
>>>>>>> go/master
required.add_argument(
    "-i",
    "--iterations",
    dest="iterations",
    type=int,
    required=True,
    help="Number of iterations per query. Must be > 1",
)
required.add_argument(
    "-m",
    "--mode",
    dest="mode",
    choices=["synthetic", "dataset"],
    required=True,
    help="Select benchmark mode. It is either synthetic or dataset. Required "
    "switches for synthetic benchmark are --synthetic-query, --num-fragments"
    " and --fragment-size. Required switches for dataset benchmark are "
    "--import-file, --table-schema-file and --queries-dir and "
    "--fragment-size is optional.",
)

# Fragment size
optional.add_argument(
    "-fs",
    "--fragment-size",
    dest="fragment_size",
    action="append",
    type=int,
    help="Fragment size to use for created table. Multiple values are allowed "
    "and encouraged. If no -fs switch is specified, default fragment size "
    "is used and templated CREATE TABLE sql files cannot be used.",
)

# Required for synthetic benchmarks
optional.add_argument(
    "-nf",
    "--num-fragments",
    dest="num_synthetic_fragments",
    help="Number of fragments to generate for synthetic benchmark. Dataset size "
    "is fragment_size * num_fragments.",
)
optional.add_argument(
    "-sq",
    "--synthetic-query",
    choices=[
        "BaselineHash",
        "MultiStep",
        "NonGroupedAgg",
        "PerfectHashMultiCol",
        "PerfectHashSingleCol",
        "Sort",
    ],
    dest="synthetic_query",
    help="Synthetic benchmark query group.",
)

# Required for traditional data benchmarks
optional.add_argument(
    "-f",
    "--import-file",
    dest="import_file",
    help="Absolute path to file or wildcard on omnisci_server machine with data "
    "for import test. If wildcard is used, all files are imported in one "
    "COPY statement. Limiting number of files is possible using curly "
    "braces wildcard, e.g. trips_xa{a,b,c}.csv.gz.",
)
optional.add_argument(
    "-c",
    "--table-schema-file",
    dest="table_schema_file",
    help="Path to local file with CREATE TABLE sql statement for " "the import table.",
)
optional.add_argument(
<<<<<<< HEAD
    "-d",
    "--queries-dir",
    dest="queries_dir",
    help="Absolute path to dir with query files.",
)

# MySQL database parameters
optional.add_argument(
    "-db-server", default="localhost", help="Host name of MySQL server."
)
optional.add_argument(
    "-db-port", default=3306, type=int, help="Port number of MySQL server."
)
=======
    "-d", "--queries-dir", dest="queries_dir", help="Absolute path to dir with query files."
)

# MySQL database parameters
optional.add_argument("-db-server", default="localhost", help="Host name of MySQL server.")
optional.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server.")
>>>>>>> go/master
optional.add_argument(
    "-db-user",
    default="",
    help="Username to use to connect to MySQL database. "
    "If user name is specified, script attempts to store results in MySQL "
    "database using other -db-* parameters.",
)
optional.add_argument(
<<<<<<< HEAD
    "-db-pass",
    default="omniscidb",
    help="Password to use to connect to MySQL database.",
)
optional.add_argument(
    "-db-name",
    default="omniscidb",
    help="MySQL database to use to store benchmark results.",
)
optional.add_argument(
    "-db-table", help="Table to use to store results for this benchmark."
)
=======
    "-db-pass", default="omniscidb", help="Password to use to connect to MySQL database."
)
optional.add_argument(
    "-db-name", default="omniscidb", help="MySQL database to use to store benchmark results."
)
optional.add_argument("-db-table", help="Table to use to store results for this benchmark.")
>>>>>>> go/master

optional.add_argument(
    "-commit",
    default="1234567890123456789012345678901234567890",
    help="Commit hash to use to record this benchmark results.",
)

args = parser.parse_args()

if args.omnisci_cwd is not None:
    server_cwd = args.omnisci_cwd
else:
    server_cwd = pathlib.Path(args.omnisci_executable).parent.parent

conda_env = CondaEnvironment(args.env_name)
print("PREPARING ENVIRONMENT")
conda_env.create(args.env_check, requirements_file=args.ci_requirements)

dataset_import_cmdline = [
    "python3",
    os.path.join(args.benchmarks_path, "run_benchmark_import.py"),
    "-u",
    args.user,
    "-p",
    args.passwd,
    "-s",
    "localhost",
    "-o",
    str(args.omnisci_port),
    "-n",
    args.name,
    "-t",
    args.import_table_name,
    "-l",
    args.label,
    f"-f={args.import_file}",
    "-c",
    args.table_schema_file,
    "-e",
    "output",
    "-v",
    "--no-drop-table-after",
]

dataset_benchmark_cmdline = [
    "python3",
    os.path.join(args.benchmarks_path, "run_benchmark.py"),
    "-u",
    args.user,
    "-p",
    args.passwd,
    "-s",
    "localhost",
    "-o",
    str(args.omnisci_port),
    "-n",
    args.name,
    "-t",
    args.import_table_name,
    "-l",
    args.label,
    "-d",
    args.queries_dir,
    "-i",
    str(args.iterations),
    "-e",
    "file_json",
    "-j",
    "benchmark.json",
    "-v",
]

synthetic_benchmark_cmdline = [
    "python3",
    os.path.join(args.benchmarks_path, "run_synthetic_benchmark.py"),
    "--user",
    args.user,
    "--password",
    args.passwd,
    "--server",
    "localhost",
    "--port",
    str(args.omnisci_port),
    "--dest_port",
    str(args.omnisci_port),
    "--name",
    args.name,
    "--table_name",
    args.import_table_name,
    "--label",
    args.label,
    "--iterations",
    str(args.iterations),
    "--print_results",
    "--query",
    args.synthetic_query,
    "--num_fragments",
    str(args.num_synthetic_fragments),
    "--data_dir",
    os.path.join(server_cwd, "data"),
    "--gpu_label",
    "CPU",
    "--result_dir",
    "synthetic_results",
]

if args.mode == "synthetic":
    if (
        args.synthetic_query is None
        or args.num_synthetic_fragments is None
        or args.fragment_size is None
    ):
        print(
            "For synthetic type of benchmark the following parameters are mandatory: "
            "--synthetic-query, --num-fragments and --fragment-size."
        )
        sys.exit(3)
    datafiles = 0
    results_file_name = os.path.join(
        args.benchmarks_path,
        "synthetic_results",
        args.label,
        "CPU",
        "Benchmarks",
        args.synthetic_query + ".json",
    )
    import_cmdline = None
    benchmark_cmdline = synthetic_benchmark_cmdline
else:
<<<<<<< HEAD
    if (
        args.import_file is None
        or args.table_schema_file is None
        or args.queries_dir is None
    ):
=======
    if args.import_file is None or args.table_schema_file is None or args.queries_dir is None:
>>>>>>> go/master
        print(
            "For dataset type of benchmark the following parameters are mandatory: --import-file,"
            " --table-schema-file and --queries-dir and --fragment-size is optional."
        )
        sys.exit(3)
    datafiles_names = list(braceexpand(args.import_file))
    datafiles_names = sorted([x for f in datafiles_names for x in glob.glob(f)])
    datafiles = len(datafiles_names)
    print("NUMBER OF DATAFILES FOUND:", datafiles)
    results_file_name = os.path.join(args.benchmarks_path, "benchmark.json")
    import_cmdline = dataset_import_cmdline
    benchmark_cmdline = dataset_benchmark_cmdline

db_reporter = None
if args.db_user is not "":
    if args.db_table is None:
        print("--db-table parameter is mandatory to store results in MySQL database")
        sys.exit(4)
    print("CONNECTING TO DATABASE")
    db = mysql.connector.connect(
        host=args.db_server,
        port=args.db_port,
        user=args.db_user,
        passwd=args.db_pass,
        db=args.db_name,
    )
    db_reporter = DbReport(
        db,
        args.db_table,
        {
            "FilesNumber": "INT UNSIGNED NOT NULL",
            "FragmentSize": "BIGINT UNSIGNED NOT NULL",
            "BenchName": "VARCHAR(500) NOT NULL",
            "BestExecTimeMS": "BIGINT UNSIGNED",
            "BestTotalTimeMS": "BIGINT UNSIGNED",
            "WorstExecTimeMS": "BIGINT UNSIGNED",
            "WorstTotalTimeMS": "BIGINT UNSIGNED",
            "AverageExecTimeMS": "BIGINT UNSIGNED",
            "AverageTotalTimeMS": "BIGINT UNSIGNED",
        },
        {"ScriptName": "run_omnisci_benchmark.py", "CommitHash": args.commit},
    )


try:
    omnisci_server = OmnisciServer(
        omnisci_executable=args.omnisci_executable,
        omnisci_port=args.omnisci_port,
        database_name=args.name,
        omnisci_cwd=args.omnisci_cwd,
        user=args.user,
        password=args.passwd,
    )
    omnisci_server.launch()

    with open(args.report, "w") as report:
        print(
            "datafiles,fragment_size,query,query_exec_min,query_total_min,query_exec_max,"
            "query_total_max,query_exec_avg,query_total_avg,query_error_info",
            file=report,
            flush=True,
        )
        if args.fragment_size is not None:
            for fs in args.fragment_size:
                print("RUNNING WITH FRAGMENT SIZE", fs)
                execute_benchmark(
                    datafiles,
                    import_cmdline,
                    args.benchmarks_path,
                    benchmark_cmdline,
                    fs,
                    results_file_name,
                    report,
                    conda_env,
                )
        else:
            print("RUNNING WITH DEFAULT FRAGMENT SIZE")
            execute_benchmark(
                datafiles,
                import_cmdline,
                args.benchmarks_path,
                benchmark_cmdline,
                None,
                results_file_name,
                report,
                conda_env,
            )
finally:
    print("TERMINATING SERVER")
    if omnisci_server:
        omnisci_server.terminate()
    if args and args.save_env is False:
        conda_env.remove()
