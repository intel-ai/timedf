Running available benchmarks
============================

We prepared various data processing benchmarks to run.


Loading data
------------

Benchmarks require preloaded datasets to run.

**TBD**

Running benchmark
--------------------------

Let's reproduce one of existing benchmarks starting from unconfigured system.
We expect you to have **activated conda environment**.
We will use ``ny_taxi_ml`` benchmark as an example.

#. Download data **TBD** and set environment variable with path to your dataset storage: ``export DATASETS_PWD="/datasets"``
#. Create new environment where you will store all dependencies ``export ENV_NAME="ny_taxi_ml" && conda create -y -n $ENV_NAME python=3.9``
#. Clone repository and install dependencies: ``git clone https://github.com/intel-ai/omniscripts.git && cd omniscripts && pip install ".[all]"``
#. *Optinal, not needed for ny_taxi_ml*. Install benchmark-specific dependencies: ``conda env -n $ENV_NAME update -f omniscripts/benchmarks/$BENCHMARK_NAME/requirements.yaml``
#. *Optinal, not needed for ny_taxi_ml*. If you want to store results in a database, define environment variable with parameters: ``export DB_COMMON_OPTS=""``. For example, to save results to local sqlite database (essentially just file on your filesystem) use ``export DB_COMMON_OPTS="-db_name db.sqlite"``
#. You can now run benchmark with pandas: ``PANDAS_MODE="Pandas" ./build_scripts/ny_taxi_ml.sh`` or modin on ray: ``PANDAS_MODE="Modin_on_ray" ./build_scripts/ny_taxi_ml.sh`` or HDK ``PANDAS_MODE="Modin_on_hdk" ./build_scripts/ny_taxi_ml.sh``

List of available benchmarks
----------------------------

Here are the exising benchmarks:

#. census - simple ETL and ML based on US census data.
#. H2O - H2O benchmark with join and groupby operations based on https://h2oai.github.io/db-benchmark/
#. hm_fashion_recs - large benchmark with complex data processing for recommendation systems based on one of the top solutions to kaggle competiton https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations 
#. ny_taxi - 4 queries (mainly gropuby) for NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page 
#. ny_taxi_ml - simple ETL and ML based on NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#. plasticc - simple ETL and ML for plasticc dataset https://plasticc.org/data-release/
#. optiver_vol - benchmark with winning solution to optiver volatility competition based on https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution/notebook

Each benchmark is stored in it's own folder in ``omniscripts/benchmarks/``

Validating intermediate dataframes
----------------------------------

You might want to validate that dataframe processing library is providing results, consistent with other libraries.
This is an optional result validation feature that is not yet available, but will be provided in the future.

**TBD**

Build scripts
-------------

There are preset configurations for existing benchmarks, located in ``build_scripts`` folder.

To run these scripts you need to define several environment variables:

1. ``PANDAS_MODE`` needs to be one of pandas modes, supported by ``run_modin_tests.py``. Currently that's: ``Pandas``, ``Modin_on_ray``, ``Modin_on_hdk``.
2. ``DATASETS_PWD`` - root of datasets storage.
3. ``ENV_NAME`` - name of the conda environment to use.

Some additional parameters are optional:

1. ``DB_COMMON_OPTS`` - should contain database parameters as supported by ``run_modin_tests.py``, if not provided no result saving will be performed. To save to local sqlite file use ``export DB_COMMON_OPTS="-db_name db.sqlite"``.
2. ``ADDITIONAL_OPTS``` - additional arguments for `benchmark-run` command.

After defining environment variables and **activating conda** you need to run command like this:
``./build_scripts/ny_taxi_ml.sh .``
Of course, you can provide some or all environment variables with a command like this:
``PANDAS_MODE="Pandas" ./build_scripts/ny_taxi_ml.sh``
