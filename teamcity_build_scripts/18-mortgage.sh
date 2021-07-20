#!/bin/bash -e

python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark        \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                      \
                          --ibis_path "${PWD}/../ibis/"                                                                       \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                          \
                          -database_name ${DATABASE_NAME} -table mg_bench_t -bench_name mortgage -dfiles_num 1 -iterations 1  \
                          -ipc_conn True -columnar_output True -lazy_fetch False -multifrag_rs True                           \
                          -fragments_size 2000000 2000000 -import_mode fsi -omnisci_run_kwargs enable-union=1                 \
                          -data_file '${DATASETS_PWD}/mortgage_new'                                                           \
                          -pandas_mode Pandas                                                                                 \
                          -optimizer intel -gpu_memory 16                                                                     \
                          -commit_omnisci ${BUILD_REVISION}                                                                   \
                          -commit_ibis ${BUILD_IBIS_REVISION}                                                                 \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                   \
                          ${ADDITIONAL_OPTS}                                                                                  \
                          ${DB_COMMON_OPTS} ${DB_MORTGAGE_OPTS}
