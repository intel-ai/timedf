#!/bin/bash -e

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
mkdir -p ${PWD}/tmp
python3 run_modin_tests.py -bench_name h2o -data_file "${DATASETS_PWD}/h2o/G1_1e7_1e2_0_0.csv"                                     \
                              -task benchmark -pandas_mode Modin_on_hdk                                                            \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS}

conda deactivate
