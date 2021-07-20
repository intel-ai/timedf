#!/bin/bash -e

python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 --task test            \
                          --ibis_path "${PWD}"/../ibis/                                                                       \
                          -executable "${PWD}"/../omniscidb/build/bin/omnisci_server                                          \
                          -database_name ${DATABASE_NAME} --report_path "${PWD}"/..                                           \
                          -commit_omnisci ${BUILD_REVISION} -commit_ibis ${BUILD_IBIS_REVISION}                               \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                   \
                          -user admin -password HyperInteractive                                                              \
                          -db_name ${DATABASE_NAME}
