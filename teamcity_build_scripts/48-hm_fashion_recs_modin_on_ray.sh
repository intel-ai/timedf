#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name hm_fashion_recs                                                             \
                          -data_file "${DATASETS_PWD}/hm_fashion_recs/"                                                               \
                          -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                                     \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_HM_FASHION_RECS_OPTS}
