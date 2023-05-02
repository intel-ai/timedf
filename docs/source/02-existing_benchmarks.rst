Running available benchmarks
============================

We have several benchmarks to run:

#. ``census`` - simple ETL and ML based on US census data.
#. ``plasticc`` - simple ETL and ML for plasticc dataset https://plasticc.org/data-release/
#. ``ny_taxi`` - 4 queries (mainly gropuby) for NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#. ``ny_taxi_ml`` - simple ETL and ML based on NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#. ``hm_fashion_recs`` - large benchmark with complex data processing for recommendation systems based on one of the top solutions to kaggle competiton https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations
#. ``optiver_vol`` - benchmark with winning solution to optiver volatility competition based on https://www.kaggle.com/code/nyanpn/1st-place-public-2nd-place-solution/notebook

..
    #. H2O - H2O benchmark with join and groupby operations based on https://h2oai.github.io/db-benchmark/

Each benchmark's source code is stored in it's own folder in ``omniscripts_benchmarks/``

Loading data
------------

To run benchmark you will first need to download the dataset::

    # This will download dataset for `$BENCHMARK` and put it into `$TARGET_DIR`
    benchmark-load $BENCHMARK $TARGET_DIR
    # For example
    benchmark-load census ./census

Running benchmark
--------------------------

Let's run one of benchmarks (``plasticc``) starting from an unconfigured system.
We expect you to have **installed omniscripts into conda environment named** ``"ENV_NAME="omniscripts"`` :ref:`installation-label`.

#. Activate your conda environment: ``conda activate $ENV_NAME``.
#. Download data ``benchmark-load plasticc ./datasets/plasticc``.
#. Run benchmark with pandas: ``benchmark-run plasticc -data_file ./datasets/plasticc -pandas_mode Pandas ${DB_COMMON_OPTS}``.
    #. To run with with modin on ray replace ``"Pandas"->"Modin_on_ray"``, for modin on HDK replace ``"Pandas"->"Modin_on_hdk"``.
    #. You can get a list of all possible parameters with ``benchmark-run -h``.
    #. *Optinal, not needed for plasticc*. You might need to install benchmark-specific dependencies with: ``conda env -n $ENV_NAME update -f omniscripts_benchmarks/$BENCHMARK_NAME/requirements.yaml``
    #. *Optinal*. If you want to store results in a database, define environment variable with parameters: ``export DB_COMMON_OPTS=""``. For example, to save results to local sqlite database (essentially just file on your filesystem) use ``export DB_COMMON_OPTS="-db_name db.sqlite"``

Validating intermediate dataframes
----------------------------------

You might want to validate that dataframe processing library is providing results, consistent with other libraries.
This is an optional result validation feature that is not yet available, but will be provided in the future.

**TBD**
