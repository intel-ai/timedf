# Omniscripts

Scripts that are used to run benchmarks for dataframe libraries, such as modin.

[Full documentation](https://omniscripts.readthedocs.io/en/latest/)
## Installation

`pip install .`

If you want to use SQL database for result storage install with `[reporting]`:
`pip install ".[reporting]"`

## Requirements

We recommend using dedicated conda environment to run benchmarks.

## Main benchmarks launching script

Main script is called `benchmark-run`. `benchmark-run -h` will show all parameters. 

Results can be stored in SQL database and visualized using Grafana charts, xlsx or any other tool capable of extracting SQL data.

The easiest way to run scripts is to use pre-written scripts from `build_scripts` folder

Also there is `docker/e2e-hdk/all.sh` which can be served as example of all steps that have to be done for benchmarks launching including report generation.


## Development

If you are a developer you might benefit from installing code style tools and testing dependencies:

`pip install -r requirements/dev_reqs.txt`

## Standalone benchmark launch

TBD
