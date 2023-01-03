import pytest

from sqlalchemy import create_engine, select
from report.report import DbReport


@pytest.fixture(scope="session")
def engine():
    return create_engine("sqlite://", future=True)


def test_dbreport(engine):
    """Returns an sqlalchemy session, and after the test tears down everything properly."""
    table_name = "tablename"
    report = DbReport(engine, table_name, ["result"], {"setting1": "param1"})

    report.submit({"result": 12})

    stmt = select(report._table)

    with engine.connect() as conn:
        results = list(conn.execute(stmt))
        assert len(results)
        assert results[0]["result"] == "12"
