from collections import defaultdict

import pytest

from sqlalchemy import create_engine, select
from report.schema import Iteration, Base
from report.report import DbReporter


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite://", future=True)


def test_dbreport(engine):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    table_name = "tablename"
    report = DbReporter(engine, benchmark='testbench', run_id=1, run_params={"setting1": "param1"})

    report.report(iteration_no=1, name2time={"q1": 1.5, 'q2': 11.2}, params=defaultdict(lambda x: 'defaultval'))

    stmt = select(Iteration)

    with engine.begin() as session:
        results = list(session.execute(stmt))
        assert len(results)
        assert results[0]["q1"] == 1.5


def test_schema(engine):
    with engine.begin() as session:
        Base.metadata.create_all(engine)
        first = session.execute(select(Iteration)).first()
