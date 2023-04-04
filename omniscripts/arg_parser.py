"""Argument parsing"""
import argparse
from dataclasses import dataclass
from typing import Union

from .pandas_backend import Backend


__all__ = ["add_sql_arguments", "prepare_parser", "DbConfig"]


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

    def create_engine(self):
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


def str_arg_to_bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Cannot recognize boolean value.")


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


def prepare_parser():
    parser = argparse.ArgumentParser(description="Run benchmarks for Modin perf testing")
    optional = parser.add_argument_group("optional arguments")
    benchmark = parser.add_argument_group("benchmark")
    sql = parser.add_argument_group("sql")
    commits = parser.add_argument_group("commits")

    # Benchmark parameters
    benchmark.add_argument("-bench_name", dest="bench_name", help="Benchmark name.")
    benchmark.add_argument(
        "-data_file", dest="data_file", help="A datafile that should be loaded."
    )
    benchmark.add_argument(
        "-dfiles_num",
        dest="dfiles_num",
        default=None,
        type=int,
        help="Number of datafiles to load into database for processing.",
    )
    benchmark.add_argument(
        "-iterations",
        dest="iterations",
        default=1,
        type=int,
        help="Number of iterations to run. All results will be submitted to the DB.",
    )
    benchmark.add_argument(
        "-validation",
        dest="validation",
        default=False,
        type=str_arg_to_bool,
        help="validate queries results (by comparison with Pandas queries results).",
    )
    benchmark.add_argument(
        "-optimizer",
        choices=["intel", "stock"],
        dest="optimizer",
        default=None,
        help="Optimizer to use.",
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
        default=None,
        type=str_arg_to_bool,
        help="Do not run machine learning benchmark, only ETL part",
    )
    benchmark.add_argument(
        "-use_modin_xgb",
        default=False,
        type=str_arg_to_bool,
        help="Whether to use Modin XGBoost for ML part, relevant for Plasticc benchmark only",
    )
    optional.add_argument(
        "-gpu_memory",
        dest="gpu_memory",
        type=int,
        help="specify the memory of your gpu"
        "(This controls the lines to be used. Also work for CPU version. )",
        default=None,
    )
    benchmark.add_argument(
        "-extended_functionality",
        dest="extended_functionality",
        default=False,
        type=str_arg_to_bool,
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
