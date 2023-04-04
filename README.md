# Scripts that are used to run modin-specific benchmarks in automated way in TeamCity and for performance analysis in development cycle.


## Installation

`pip install .`

If you want to use SQL database for result storage install with `[reporting]`:
`pip install .[reporting]`

## Requirements

Scripts require miniconda to be installed.

Library requirements are located in `requirements` folder, that is mainly `requirements/base.txt` and `requirements/reporting.txt` (for reporting).

Or you can install all requirements with: `pip install -e .[all]`

## Main benchmarks launching script

Main script is called `run_modin_tests.py`. ```run_modin_tests.py -h``` will show all script's parameters. 
Script automatically creates conda environment if it doesn't exist or you want to recreate it.
All subsequent work is being done in created conda environment. Environment can be removed or saved after executing.

Results can be stored in SQL database and visualized using Grafana charts, xlsx or any other tool capable of extracting SQL data.

The easiest way to run scripts is to use pre-written scripts from `build_scripts` folder

Also there is `docker/e2e-hdk/all.sh` which can be served as example of all steps that have to be done for benchmarks launching including report generation.

## Standalone benchmark launch

TBD
