from typing import Dict, Any, Union, Iterable, Pattern

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from report.run_params import RunParams, HostParams
from report.schema import Iteration, Measurement


class DBConfig:
    driver: str
    server: str
    port: int
    user: str
    password: str
    name: str

    def create_engine(self) -> Engine:
        url = f"{self.driver}://{self.user}:{self.password}@{self.server}:{self.port}/{self.name}"
        return create_engine(url, future=True)

class DBReporter:
    def __init__(self, db_config: DBConfig):
        """Initialize and submit reports to MySQL database

        Parameters
        ----------
        db_config
            Database engine from sqlalchemy
        table_name
            Table name
        benchmark_specific_col2sql_type
            Declaration of types that will be submitted during benchmarking along with type
            information. For example {'load_data': 'BIGINT UNSIGNED'}.
        predefined_col2value, optional
            Values that are knows before starting the benchmark, they will be submitted along with
            benchmark results, we assume string type for values.
        """
        self.engine = db_config.create_engine()

    def report(self, run_id, run_params, results, params=None):
        params = params or {}
        with Session(self.engine, autocommit=True) as session:
            measurements = [Measurement(**results) for r in results]
            iteration = Iteration(
                run_id=run_id,
                params=params,
                **HostParams().report(),
                **RunParams().report(run_params),
                measurements=measurements
            )
            session.add(iteration)
            session.commit()
