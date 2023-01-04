import pytest

from sqlalchemy import create_engine, select
from report.schema import Iteration, Base
from report.report import DbReporter


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite://", future=True)


@pytest.mark.skip("Outdated test")
def test_dbreport(engine):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    table_name = "tablename"
    report = DbReporter(engine, table_name, ["result"], {"setting1": "param1"})

    report.submit({"result": 12})

    stmt = select(report._table)

    with engine.begin() as session:
        results = list(session.execute(stmt))
        assert len(results)
        assert results[0]["result"] == "12"


def test_schema(engine):
    with engine.begin() as session:
        Base.metadata.create_all(engine)
        first = session.execute(select(Iteration)).first()
        print(first)
