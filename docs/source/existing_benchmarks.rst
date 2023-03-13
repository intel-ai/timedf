Existing benchmarks
===================

Here are the exising benchmarks:

#. census - simple ETL and ML based on US census data.
#. H2O - H2O benchmark with join and groupby operations https://h2oai.github.io/db-benchmark/
#. hm_fashion_recs - large benchmark with complex data processing for recommendation systems based on one of the top solutions to kaggle competiton https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations 
#. ny_taxi - 4 queries (mainly gropuby) for NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page 
#. ny_taxi_ml - simple ETL and ML based on NY taxi dataset https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#. plasticc - simple ETL and ML for plasticc dataset https://plasticc.org/data-release/

Build scripts
-------------

There are preset configurations for existing benchmarks, located in ``build_scripts`` folder.

To run these scripts you need to define several environment variables:
1. ``PANDAS_MODE`` needs to be one of pandas modes, supported by run_modin_tests.py . Currently that's ``Pandas``, ``Modin_on_ray``, ``Modin_on_hdk``
2. ``DATASETS_PWD``` - root of datasets.
3. ``ENV_NAME``` - name of the conda environment to use.

Some additional parameters are optional:
1. ``DB_COMMON_OPTS`` - should contain database parameters as supported by ``run_modin_tests.py`` , if not provided no result saving will be performed.
2. ``ADDITIONAL_OPTS``` - additonal arguments for `run_modin_tests.py`

After defining environment variables and **activating conda** you need to run command like this:
``./build_scripts/ny_taxi_ml.sh .``
Of course, you can provide some or all environment variables with a command like this:
``PANDAS_MODE="Pandas" ./build_scripts/ny_taxi_ml.sh``
