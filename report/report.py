from dataclasses import dataclass
from typing import Dict, Union

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from report.schema import make_iteration


# This can be written as just a function, but we keep the dataclass to add validation and arg parsing in the future.
@dataclass
class DbConfig:
    """Class encapsulates DB configuration and connection."""

    driver: str
    server: str
    port: int
    user: str
    password: str
    name: str

    def create_engine(self) -> Engine:
        url = f"{self.driver}://{self.user}:{self.password}@{self.server}:{self.port}/{self.name}"
        return create_engine(url, future=True)


class DbReporter:
    def __init__(self, engine: Engine, benchmark: str, run_id: int, run_params: Dict[str, str]):
        """Initialize and submit reports to a database

        Parameters
        ----------
        db_config
            database configuration
        benchmark
            Name of the current benchmark
        run_id
            Unique id for the current run that will contain several iterations with results
        run_params
            Parameters of the current run, reporter will extract params that are relevant for
            reporting, full list necessary params is available in RunParams class. If some of the
            fields are missing, error will be reported, extra parameters will be ignored.
        """
        self.engine = engine
        self.benchmark = benchmark
        self.run_id = run_id
        self.run_params = run_params

    def report(
        self, iteration_no: int, name2time: Dict[str, float], params: Union[None, Dict] = None
    ):
        """Report results of current iteration.

        Parameters
        ----------
        iteration_no
            Iteration number for the report
        name2time
            Dict with measurements: (name, time in seconds)
        params
            Additional params to report, will be added to a schemaless `params` column in the DB, can be used for
            storing benchmark-specific infomation such as datset size.
        """
        with Session(self.engine, autocommit=True) as session:
            session.add(
                make_iteration(
                    run_id=self.run_id,
                    benchmark=self.benchmark,
                    iteration_no=iteration_no,
                    run_params=self.run_params,
                    name2time=name2time,
                    params=params,
                )
            )
