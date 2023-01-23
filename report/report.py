from typing import Dict, List, Tuple, Union

import pandas as pd
from sqlalchemy import sql
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import datetime as dt

from report.schema import make_iteration, Base, Iteration as Iter, Measurement as M


class Db:
    def __init__(self, engine: Engine):
        """Database with benchmark results

        Parameters
        ----------
        engine
            DB engine from sqlalchemy
        """
        self.engine = engine
        Base.metadata.create_all(engine, checkfirst=True)

    def report(
        self,
        benchmark: str,
        run_id: int,
        run_params: Dict[str, str],
        iteration_no: int,
        name2time: Dict[str, float],
        params: Union[None, Dict] = None,
    ):
        """Report results of current iteration.

        Parameters
        ----------
        benchmark
            Name of the current benchmark
        run_id
            Unique id for the current run that will contain several iterations with results
        run_params
            Parameters of the current run, reporter will extract params that are relevant for
            reporting, full list necessary params is available in RunParams class. If some of the
            fields are missing, error will be reported, extra parameters will be ignored.
        iteration_no
            Iteration number for the report
        name2time
            Dict with measurements: (name, time in seconds)
        params
            Additional params to report, will be added to a schemaless `params` column in the DB, can be used for
            storing benchmark-specific information such as dataset size.
        """
        with Session(self.engine) as session:
            session.add(
                make_iteration(
                    run_id=run_id,
                    benchmark=benchmark,
                    iteration_no=iteration_no,
                    run_params=run_params,
                    name2time=name2time,
                    params=params,
                )
            )
            session.commit()

    def load_benchmarks(self, node):
        """Load a list of all benchmarks that are contained in the DB."""
        qry = sql.select(sql.func.distinct(Iter.benchmark).label("benchmark")).where(
            self._get_filter_qry(node=node)
        )

        with Session(self.engine) as session:
            return [row[0] for row in session.execute(qry).all()]

    @staticmethod
    def _get_filter_qry(benchmark=None, node=None, lookup_days=None):
        lookup_cnd = (
            True
            if lookup_days is None
            else Iter.date > (dt.date.today() - dt.timedelta(days=lookup_days))
        )

        node_cnd = True if node is None else Iter.node == node
        benchmark_cnd = True if benchmark is None else Iter.benchmark == benchmark

        return sql.and_(lookup_cnd, node_cnd, benchmark_cnd)

    def _load_qry(self, qry):
        return pd.read_sql(qry, con=self.engine, parse_dates=["date"])

    def load_iterations(self, benchmark=None, node=None, lookup_days=None):
        """Load all iterations, which satify requirements from input params."""
        qry = sql.select(Iter).filter(
            self._get_filter_qry(benchmark=benchmark, node=node, lookup_days=lookup_days)
        )
        return self._load_qry(qry).set_index("id", drop=True)

    def load_measurements(self, iteration_ids):
        """Load all measurements for selected iteration ids in wide form."""
        qry = sql.select(M).filter(M.iteration_id.in_(iteration_ids))
        df = pd.read_sql(qry, con=self.engine)
        return df.pivot(columns="name", values="duration_s", index="iteration_id")

    @staticmethod
    def add_params(df):
        """Add `parms` column content as new columns. So it's better to call it with only type of benchmark."""
        params_df = pd.DataFrame(list(df.params), index=df.index)
        return df.drop(['params'], axis=1).join(params_df)


    def load_benchmark_results(self, benchmark, node=None) -> Tuple[pd.DataFrame, List]:
        """Load benchmark results for selected `benchmark` in a wide form
        
        Returns
        -------
        df:
            DataFrame in a wide form with results
        measurements:
            List with all the measurements, contained in the returned df, so 
            `df[measurements]` is a wide table with all measurements.
        """
        df_runs = self.add_params(self.load_iterations(benchmark=benchmark, node=node))
        df_measurements = self.load_measurements(iteration_ids=list(df_runs.index))
        df = df_runs.join(df_measurements)
        return df, list(df_measurements.columns)


    def load_benchmark_results_agg(self, benchmark, node=None) -> Tuple[pd.DataFrame, List]:
        """Load benchmark results for selected `benchmark` in a wide form after aggregating 
        by run_id by taking minimum value
        
        Returns
        -------
        df:
            DataFrame in a wide form with results
        measurements:
            List with all the measurements, contained in the returned df, so 
            `df[measurements]` is a wide table with all measurements.
        """
        df, measurements = self.load_benchmark_results(benchmark=benchmark, node=node)
        res = df.groupby("run_id").agg(
            {c: "min" if c in measurements else 'first' for c in df.columns}
        )
        return res, measurements
