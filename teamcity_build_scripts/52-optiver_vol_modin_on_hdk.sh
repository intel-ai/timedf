#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name optiver_vol                                                             \
                          -data_file "${DATASETS_PWD}/optiver_realized_volatility/"                                                               \
                          -pandas_mode Modin_on_hdk -ray_tmpdir ${PWD}/tmp                                                     \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS}
