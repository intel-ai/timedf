#!/bin/bash -e

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
mkdir -p ${PWD}/tmp
python3 run_modin_tests.py -bench_name h2o -data_file "${DATASETS_PWD}/h2o/J1_1e8_NA_0_0.csv"                                      \
                              -task benchmark -pandas_mode Modin_on_hdk                                                            \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS} ${DB_H2O_OPTS}

conda deactivate
