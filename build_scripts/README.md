This folder containes preconfigured run scripts for key benchmarks. 

To run these scripts you need to define several environment variables:
1. `PANDAS_MODE` needs to be one of pandas modes, supported by `run_modin_tests.py`. Currently that's `Pandas`, `Modin_on_ray`, `Modin_on_hdk`
2. `DATASETS_PWD` - root of datasets.
3. `ENV_NAME` - name of the conda environment to use.

Some additional parameters are optional:
1. `DB_COMMON_OPTS` - should contain database parameters as supported by `run_modin_tests.py`, if not provided no result saving will be performed.
2. `ADDITIONAL_OPTS` - additonal arguments for `run_modin_tests.py`

After defining environment variables you need to **activate base conda** and then run command like this: `./build_scripts/ny_taxi_ml.sh`. Of course, you can provide some or all environment variables with a command like this: `PANDAS_MODE="Pandas" ./build_scripts/ny_taxi_ml.sh`
