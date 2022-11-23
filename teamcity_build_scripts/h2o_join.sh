#!/bin/bash

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark         \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -bench_name h2o -data_file '/localdisk/izamyati/modin/h2o/J1_1e7_NA_0_0.csv'                         \
                          -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                                     \
                          --modin_path "${PWD}/../modin"                                                                       \
                          -commit_hdk 123                                                                                      \
                          -commit_omniscripts 567                                                                              \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_SANTANDER_OPTS}
