#!/bin/bash -e

python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name h2o -data_file "${H2O_GROUPBY_TEST_DATASET}" -pandas_mode Pandas                         \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS}
