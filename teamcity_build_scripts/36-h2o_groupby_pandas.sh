#!/bin/bash -e

python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                              \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -bench_name h2o -data_file '${H2O_GROUPBY_TEST_DATASET}' -pandas_mode Pandas                         \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_H2O_OPTS}
