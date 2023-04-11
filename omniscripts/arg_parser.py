"""Argument parsing"""
import argparse
from dataclasses import dataclass
from typing import Callable

from .pandas_backend import Backend


__all__ = ["add_sql_arguments", "prepare_general_parser", "parse_args", "DbConfig"]


# This can be written as just a function, but we keep the dataclass to add validation and arg parsing in the future.
@dataclass
class DbConfig:
    """Class encapsulates DB configuration and connection.

    For the sqlite you need to pass the path to the file as `name` argument, like
    `name='database.db', driver='sqlite+pysqlite'`.
    """

    driver: str
    server: str = None
    port: int = None
    user: str = None
    password: str = None
    name: str = None

    def is_config_complete(self):
        return self.user is not None

    def _create_engine(self):
        from sqlalchemy import create_engine
        from sqlalchemy.engine.url import URL

        url = URL.create(
            host=self.server,
            drivername=self.driver,
            username=self.user,
            password=self.password,
            port=self.port,
            database=self.name,
        )
        return create_engine(url)

    def maybeCreateBenchmarkDb(self):
        if self.is_config_complete():
            from .report import BenchmarkDb

            return BenchmarkDb(self._create_engine())
        else:
            return None


def add_sql_arguments(parser):
    parser.add_argument(
        "-db_driver",
        dest="db_driver",
        default="sqlite+pysqlite",
        help="Driver for the sql table in sqlalchemy format.",
    )
    parser.add_argument("-db_server", dest="db_server", help="Host name of SQL server.")
    parser.add_argument("-db_port", dest="db_port", type=int, help="Port number of SQL server.")
    parser.add_argument(
        "-db_user",
        dest="db_user",
        help="Username to use to connect to SQL database. "
        "If user name is specified, script attempts to store results in SQL "
        "database using other -db-* parameters.",
    )
    parser.add_argument(
        "-db_pass",
        dest="db_pass",
        help="Password to use to connect to SQL database.",
    )
    parser.add_argument(
        "-db_name",
        dest="db_name",
        help="SQL database to use to store benchmark results.",
    )


def prepare_general_parser():
    parser = argparse.ArgumentParser(description="Run benchmarks for Modin perf testing")
    optional = parser.add_argument_group("optional arguments")
    benchmark = parser.add_argument_group("benchmark")
    sql = parser.add_argument_group("sql")
    commits = parser.add_argument_group("commits")

    # Benchmark parameters
    benchmark.add_argument("bench_name", help="Benchmark name.")
    benchmark.add_argument(
        "-data_file", dest="data_file", help="A datafile that should be loaded.", required=True
    )
    benchmark.add_argument(
        "-iterations",
        dest="iterations",
        default=1,
        type=int,
        help="Number of iterations to run. All results will be submitted to the DB.",
    )
    benchmark.add_argument(
        "-pandas_mode",
        choices=Backend.supported_backends,
        default="Pandas",
        help="Specifies which backend to use: "
        "plain Pandas, Modin runing on Ray or on Dask or on HDK",
    )
    benchmark.add_argument(
        "-ray_tmpdir",
        default="/tmp",
        help="Location where to keep Ray plasma store. "
        "It should have enough space to keep -ray_memory",
    )
    benchmark.add_argument(
        "-ray_memory",
        default=200 * 1024 * 1024 * 1024,
        type=int,
        help="Size of memory to allocate for Ray plasma store",
    )
    benchmark.add_argument(
        "-no_ml",
        default=False,
        action="store_true",
        help="Do not run machine learning benchmark, only ETL part",
    )
    benchmark.add_argument(
        "-use_modin_xgb",
        default=False,
        action="store_true",
        help="Whether to use Modin XGBoost for ML part, relevant for Plasticc benchmark only",
    )
    benchmark.add_argument(
        "-extended_functionality",
        dest="extended_functionality",
        default=False,
        action="store_true",
        help="Extends functionality of H2O benchmark by adding 'chk' functions and verbose local reporting of results",
    )
    # SQL database parameters
    add_sql_arguments(sql)
    # Additional information
    commits.add_argument(
        "-commit_hdk",
        dest="commit_hdk",
        default="1234567890123456789012345678901234567890",
        help="HDK commit hash used for tests.",
    )
    commits.add_argument(
        "-commit_omniscripts",
        dest="commit_omniscripts",
        default="1234567890123456789012345678901234567890",
        help="Omniscripts commit hash used for tests.",
    )
    commits.add_argument(
        "-commit_modin",
        dest="commit_modin",
        default="1234567890123456789012345678901234567890",
        help="Modin commit hash used for tests.",
    )
    return parser


def parse_args(add_benchmark_args: Callable[[argparse.ArgumentParser], None]):
    """Parse arguments including benchmark-specific arguments that will be added using provided
    `add_benchmark_args` callable"""
    parser = prepare_general_parser()
    benchmark_parser = parser.add_argument_group("benchmark_specific")
    add_benchmark_args(benchmark_parser)
    args = parser.parse_args()

    db_config = DbConfig(
        driver=args.db_driver,
        server=args.db_server,
        port=args.db_port,
        user=args.db_user,
        password=args.db_pass,
        name=args.db_name,
    )

    return args, db_config
