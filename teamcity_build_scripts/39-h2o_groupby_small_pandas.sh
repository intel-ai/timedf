#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name h2o -data_file "${DATASETS_PWD}/h2o/G1_1e7_1e2_0_0.csv"                                  \
                          -pandas_mode Pandas -ray_tmpdir ${PWD}/tmp                                                           \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS}
