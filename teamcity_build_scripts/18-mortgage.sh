#!/bin/bash -e

python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark        \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                      \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                          \
                          -data_file '${DATASETS_PWD}/mortgage_new'                                                           \
                          -pandas_mode Pandas                                                                                 \
                          -optimizer intel -gpu_memory 16                                                                     \
                          -commit_omnisci ${BUILD_REVISION}                                                                   \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                   \
                          ${ADDITIONAL_OPTS}                                                                                  \
                          ${DB_COMMON_OPTS} ${DB_MORTGAGE_OPTS}
