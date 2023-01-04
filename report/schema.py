from typing import Dict, List
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

from report.run_params import RunParams, HostParams

Base = declarative_base()


STRING_LENGTH = 200


def make_string():
    return Column(String(STRING_LENGTH), nullable=False)


# Equivalent of defining a class, but with variable class attributes
Iteration = type(
    "Iteration",
    (Base,),
    {
        "__tablename__": "iteration",
        # Iteration id
        "id": Column(Integer, primary_key=True),
        # Iteration counter
        "iteration_no": Column(Integer, nullable=False),
        # Run id, each run contains 1 or more iterations
        "run_id": Column(Integer, nullable=False),
        # date of the current iteration
        "date": Column(DateTime(), nullable=False, server_default=func.now()),
        "measurements": relationship("Measurement", back_populates="iteration"),
        # host info
        **{name: make_string() for name in HostParams.fields},
        # run params
        **{name: make_string() for name in RunParams.fields},
        # Additional params without forced schema
        "params": Column(JSON),
    },
)


class Measurement(Base):
    __tablename__ = "measurement"
    id = Column(Integer, primary_key=True)
    # Name of the measurement
    name = Column(String(STRING_LENGTH), nullable=False)
    # Duration in seconds
    duration_s = Column(Float, nullable=False)

    iteration_id = Column(Integer, ForeignKey("iteration.id"))
    iteration = relationship("Iteration", back_populates="measurements")

    # Additional data without forced schema
    params = Column(JSON)


def make_iteration(
    run_id: int, iteration_no: int, run_params, name2time: Dict[str, float], params=None
) -> Iteration:
    measurements_orm = [Measurement(name=name, duration_s=time) for name, time in name2time.items()]
    return Iteration(
        run_id=run_id,
        iteration_no=iteration_no,
        params=params,
        **HostParams().report(),
        **RunParams().report(run_params),
        measurements=measurements_orm,
    )
