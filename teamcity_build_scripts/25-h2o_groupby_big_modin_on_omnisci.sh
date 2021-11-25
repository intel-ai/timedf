#!/bin/bash -e

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
mkdir -p ${PWD}/tmp
python3 run_modin_benchmark.py -bench_name h2o -data_file "${DATASETS_PWD}/h2o/G1_1e9_1e2_0_0.csv"                                  \
                              -pandas_mode Modin_on_omnisci -ray_tmpdir ${PWD}/tmp                                                 \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS} ${DB_H2O_OPTS}

conda deactivate
