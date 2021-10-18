#!/bin/bash -e

python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark         \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                           \
                          -user admin -password HyperInteractive                                                               \
                          -database_name ${DATABASE_NAME} -table santander -bench_name santander -dfiles_num 1 -iterations 1   \
                          -data_file '${DATASETS_PWD}/santander/train.csv.gz'                                                  \
                          -pandas_mode Pandas -ray_tmpdir /tmp                                                                 \
                          -optimizer intel -gpu_memory 16                                                                      \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${DB_COMMON_OPTS} ${DB_SANTANDER_OPTS}
