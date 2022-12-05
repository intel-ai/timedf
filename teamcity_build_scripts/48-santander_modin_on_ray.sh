#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name santander                                                                                \
                          -data_file "${DATASETS_PWD}/santander/train.csv"                                                     \
                          -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                                     \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_SANTANDER_OPTS}
