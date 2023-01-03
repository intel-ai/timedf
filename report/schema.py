from sqlalchemy import Column, DateTime, String, Float, Integer, ForeignKey, JSON, func, MetaData, Table
from sqlalchemy.orm import declarative_base, relationship, registry

from report.run_params import RunParams, HostParams

Base = declarative_base()


STRING_LENGTH = 200



def make_string():
    return Column(String(STRING_LENGTH), nullable=False)
    

# Class attributes for Iteration class
_iteration_attrs = {
    '__tablename__': 'iteration',
    # Iteration id
    'id': Column(Integer, primary_key=True),
    # Iteration counter
    'iteration_no': Column(Integer, nullable=False),
    # Run id, each run contains 1 or more iterations
    'run_id': Column(Integer, nullable=False),
    # date of the current iteration
    'date': Column(DateTime(), nullable=False, server_default=func.now()),
    'measurements': relationship('Measurement', back_populates="iteration"),
    # host info
    **{name: make_string() for name in HostParams.fields},
    # run params
    **{name: make_string() for name in RunParams.fields},
    # Additional params without forced schema
    'params': Column(JSON),
}

def report_iteration(date, run_params, commit_params, other_params):
    pass

# Equivalent of defining a class, but with veriable class attributes
Iteration = type('Iteration', (Base,), _iteration_attrs)



class Measurement(Base):
    __tablename__ = "measurement"
    id = Column(Integer, primary_key=True)
    # Name of the measurement
    name = Column(String(STRING_LENGTH), nullable=False)
    # Duration in seconds
    duration_s = Column(Float, nullable=False)

    iteration_id = Column(Integer, ForeignKey('iteration.id'))
    iteration = relationship('Iteration', back_populates="measurements")

    # Additional data without forced schema
    params = Column(JSON)
