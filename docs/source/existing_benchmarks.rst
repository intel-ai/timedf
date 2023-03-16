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

Running existing benchmark
--------------------------

Let's reproduce one of existing benchmarks from unconfigured system. We expect you to have **activated conda environment**. We will use ``hm_fashion_recs.full``.

#. ``git clone https://github.com/intel-ai/omniscripts.git && cd omniscripts``
#. Create new environment where you will store all dependencies ``export ENV_NAME="hm"``
#. Install omniscripts dependencies: ``conda create -y -n $ENV_NAME && conda activate $ENV_NAME && conda env update -f requirements/base.yml && conda env update -f requirements/reporting.yml``
#. Install benchmark-specific dependencies: ``conda env update -f benchmarks/hm_fashion_recs/requirements.yaml``
#. Install latest modin ``pip install "modin[all] @ git+https://github.com/modin-project/modin"``
#. You need to download hm_fashion_recs dataset. You can do that from kaggle using the link from benchmark readme, located in ``benchmarks/hm_fashion_recs``. After downloading it, store it in a folder named ``hm_fashion_recs`` and store it's parent's location to an environment variable: ``export DATASETS_PWD="/datasets"``. In this example dataset itself is stored like this: ``/datasets/hm_fashion_recs/articles.csv``.
#. Deactivate current conda env: ``conda deactivate``
#. If you want to store results in a database, define environment variable with parameters: ``export DB_COMMON_OPTS=""``
#. You can now run benchmark with pandas: ``PANDAS_MODE="Pandas" ./build_scripts/hm_fashion_recs_full.sh`` or modin on ray: ``PANDAS_MODE="Modin_on_ray" ./build_scripts/hm_fashion_recs_full.sh`` or HDK ``PANDAS_MODE="Modin_on_hdk" ./build_scripts/hm_fashion_recs_full.sh``
