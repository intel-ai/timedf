#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                              \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -bench_name h2o -data_file '${DATASETS_PWD}/h2o/G1_1e9_1e2_0_0.csv'                                  \
                          -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                                     \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          -commit_modin ${BUILD_MODIN_REVISION}                                                                \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_H2O_OPTS}
