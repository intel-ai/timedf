from typing import Dict

from sqlalchemy import (
    Column,
    DateTime,
    String,
    Float,
    Integer,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

from .run_params import RunParams, HostParams

Base = declarative_base()


STRING_LENGTH = 200
LARGE_STRING_LENGTH = 500


def make_string(nullable=False):
    return Column(String(STRING_LENGTH), nullable=nullable)


# Equivalent of defining a class, but with variable class attributes
Iteration = type(
    "Iteration",
    (Base,),
    {
        "__tablename__": "iteration",
        # Iteration id
        "id": Column(Integer, primary_key=True),
        # Name of the benchmark
        "benchmark": make_string(),
        "backend": make_string(),
        # Iteration counter
        "iteration_no": Column(Integer, nullable=False),
        # Run id, each run contains 1 or more iterations
        "run_id": Column(Integer, nullable=False),
        # date of the current iteration
        "date": Column(DateTime(), nullable=False, server_default=func.now()),
        "measurements": relationship("Measurement", back_populates="iteration"),
        "checksums": relationship("Checksum", back_populates="iteration"),
        # host info
        **{name: make_string() for name in HostParams.fields},
        # run params
        **{name: make_string(nullable=True) for name in RunParams.fields},
        # Backend params
        "backend_params": Column(JSON, nullable=False),
        # Additional params without forced schema
        "params": Column(JSON),
    },
)


class Measurement(Base):
    __tablename__ = "measurement"
    id = Column(Integer, primary_key=True)
    # Name of the measurement
    name = Column(String(LARGE_STRING_LENGTH), nullable=False)
    # Duration in seconds
    duration_s = Column(Float, nullable=False)

    iteration_id = Column(Integer, ForeignKey("iteration.id"))
    iteration = relationship("Iteration", back_populates="measurements")

    # Additional data without forced schema
    params = Column(JSON)


class Checksum(Base):
    __tablename__ = "checksum"
    id = Column(Integer, primary_key=True)
    # Name of the checksum
    name = Column(String(LARGE_STRING_LENGTH), nullable=False)
    # Value of this checksum
    value = Column(Float, nullable=False)
    # Duration in seconds
    duration_s = Column(Float, nullable=False)

    iteration_id = Column(Integer, ForeignKey("iteration.id"))
    iteration = relationship("Iteration", back_populates="checksums")


def make_iteration(
    run_id: int,
    benchmark: str,
    backend: str,
    iteration_no: int,
    run_params,
    name2time: Dict[str, float],
    checksums: Dict[str, Dict],
    backend_params,
    params=None,
) -> Iteration:
    measurements_orm = [
        Measurement(name=name, duration_s=time) for name, time in name2time.items()
    ]
    checksums_orm = [
        Checksum(name=k, duration_s=float(v["duration_s"]), value=float(v["value"]))
        for k, v in checksums.items()
    ]
    return Iteration(
        run_id=run_id,
        benchmark=benchmark,
        backend=backend,
        iteration_no=iteration_no,
        params=params,
        backend_params=backend_params,
        **HostParams().prepare_report_dict(),
        **RunParams().prepare_report_dict(run_params),
        measurements=measurements_orm,
        checksums=checksums_orm,
    )
