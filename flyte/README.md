## Benchmarking scripts that are used to run Flytekit benchmarks for performance analyzes in development cycle.

## There are implemented following benchmarks:
* taxi
* census
* plasticc
* santander

## Requirements
Scripts require to be installed:
* Git and python >= 3.7, pip3
* conda or miniconda for flytekit tests and benchmarks;
* the following python3 packages: flytekit>=0.20.1.

## Flytekit installation
`(venv)$ pip3 install flytekit --upgrade`

## Running benchmarks instructions
* copy and open **jupyter/** .ipynb scripts via `jupyter notebook`
* or open  **scrips/ .py** in text redactor
* replace default path in the input variables in the function decorated as `@workflow`
* run sequentially notebook cells or run function scripts as `(venv) $ python <benchmark_file_name>`
