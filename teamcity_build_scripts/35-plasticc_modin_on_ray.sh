#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name plasticc                                                                                 \
                          -data_file "${DATASETS_PWD}/plasticc/"                                                               \
                          -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                                     \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_PLASTICC_OPTS}
