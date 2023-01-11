import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from report.schema import Iteration, Base
from report.report import DbReporter
from report.run_params import RunParams


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite://", future=True)


def test_schema(engine):
    Base.metadata.create_all(engine)


def test_dbreport(engine):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    report = DbReporter(
        engine,
        benchmark="testbench",
        run_id=1,
        run_params={k: "testval" for k in RunParams.fields},
    )

    # with Session(engine, autocommit=True) as session:
    Base.metadata.create_all(engine)

    report.report(
        iteration_no=1,
        name2time={"q1": 1.5, "q2": 11.2},
        params={k: "testval" for k in RunParams.fields},
    )

    with Session(engine) as session:
        stmt = select(Iteration)
        results = list(session.execute(stmt).scalars().all())
        assert len(results) == 1
        assert len(results[0].measurements) == 2
