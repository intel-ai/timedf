Running available benchmarks
============================

We have several benchmarks to run:

#. ``plasticc`` - simple ETL and ML for plasticc dataset https://plasticc.org/data-release/
#. ``ny_taxi`` - 4 queries (mainly gropuby) for NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#. ``ny_taxi_ml`` - simple ETL and ML based on NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#. ``hm_fashion_recs`` - large benchmark with complex data processing for recommendation systems based on one of the top solutions to kaggle competiton https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations

Each benchmark's source code is stored in it's own folder in ``timedf_benchmarks/``

Loading data
------------

To run benchmark you will first need to download the dataset::

    # This will download dataset for `$BENCHMARK` and put it into `$TARGET_DIR`
    benchmark-load $BENCHMARK $TARGET_DIR
    # For example
    benchmark-load census ./census

Running benchmark
--------------------------

Let's run one of benchmarks (``plasticc``) starting from a system with :ref:`installed timedf in conda environment <installation-label>` named ``ENV_NAME="timedf"``.

#. Activate your conda environment: ``export ENV_NAME="timedf" && conda activate $ENV_NAME``.
#. Download data ``benchmark-load plasticc ./datasets/plasticc``.
#. Run benchmark with pandas: ``benchmark-run plasticc -data_file ./datasets/plasticc -backend Pandas ${DB_COMMON_OPTS}``.
    #. To run with with modin on ray replace ``"Pandas"->"Modin_on_ray"``, for modin on HDK replace ``"Pandas"->"Modin_on_hdk"``.
    #. You can get a list of all possible parameters with ``benchmark-run -h``.
    #. *Optinal, not needed for plasticc*. You might need to install benchmark-specific dependencies with: ``conda env -n $ENV_NAME update -f timedf_benchmarks/$BENCHMARK_NAME/requirements.yaml``
    #. *Optinal*. If you want to store results in a database, define environment variable with parameters: ``export DB_COMMON_OPTS=""``. For example, to save results to local sqlite database (essentially just file on your filesystem) use ``export DB_COMMON_OPTS="-db_name db.sqlite"``


If you want to customize name of your backend or name of your benchmark you can use these arguments:

#. ``-save_benchmark_name BENCHMARK_NAME`` - benchmark name for DB storage
#. ``-save_backend_name BACKEND_NAME`` - name of the backend used for DB storage. You can use this for experimental branches of libraries.

Validating intermediate dataframes
----------------------------------

You might want to validate that dataframe processing library is providing results, consistent with other libraries.
This is an optional result validation feature that is not yet available, but will be provided in the future.

**TBD**
