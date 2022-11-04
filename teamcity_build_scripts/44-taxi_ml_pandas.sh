#!/bin/bash -e

ENV_NAME=taxi
DATASETS_PWD=/localdisk/benchmark_datasets
DB_COMMON_OPTS="-db_server ansatlin07.an.intel.com -db_port 3306 -db_user gashiman -db_pass omniscidb -db_name omniscidb"
DB_TAXIML_OPTS="-db_table_etl taximl_etl_jit"

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name taxi_ml -use_modin_xgb False                                                             \
                          -data_file "${DATASETS_PWD}/yellow-taxi-dataset/"                                                               \
                          -pandas_mode Pandas -ray_tmpdir ${PWD}/tmp                                                     \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_TAXIML_OPTS}
