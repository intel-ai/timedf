#!/bin/bash -e

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
python3 run_modin_tests.py -bench_name santander -data_file "${DATASETS_PWD}/santander/train.csv"                                  \
                              -task benchmark -pandas_mode Modin_on_hdk                                                            \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS} ${DB_SANTANDER_OPTS}

conda deactivate
