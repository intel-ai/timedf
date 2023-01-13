#!/bin/bash -e

# Modin xgb doesn't support HDK backend
mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name taxi_ml -use_modin_xgb False                                                             \
                          -data_file "${DATASETS_PWD}/yellow-taxi-dataset/"                                                    \
                          -pandas_mode Modin_on_hdk                                                                            \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS}
