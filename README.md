[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

# timedf

Scripts that are used to run benchmarks for dataframe libraries, such as modin.

[Full documentation](https://timedf.readthedocs.io/en/latest/)

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

## Datasets

To the extent that any public datasets are referenced by Intel or accessed using tools or code on this site those datasets are provided by the third party indicated as the data source. Intel does not create the data, or datasets, and does not warrant their accuracy or quality. By accessing the public dataset(s) you agree to the terms associated with those datasets and that your use complies with the applicable license. [DATASETS](DATASETS.md)

Intel expressly disclaims the accuracy, adequacy, or completeness of any public datasets, and is not liable for any errors, omissions, or defects in the data, or for any reliance on the data. Intel is not liable for any liability or damages relating to your use of public datasets.
