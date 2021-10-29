#!/bin/bash -e

python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark       \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                          \
                          -data_file '${DATASETS_PWD}/mortgage_new'                                                           \
                          -pandas_mode Pandas                                                                                 \
                          -optimizer intel -gpu_memory 16                                                                     \
                          ${ADDITIONAL_OPTS}                                                                                  \
                          ${DB_COMMON_OPTS} ${DB_MORTGAGE_OPTS}
