#!/bin/bash -e

BENCH_NAME="h2o"
DATA_FILE="${DATASETS_PWD}/h2o/J1_1e7_NA_0_0.csv" 

python3 run_modin_tests.py -task benchmark                                   \
                           -bench_name h2o                                   \
                           -data_file "${DATA_FILE}"                         \
                           -pandas_mode "${PANDAS_MODE}"                     \
                           -ray_tmpdir ${PWD}/tmp                            \
                           ${ADDITIONAL_OPTS}                                \
                           ${ADDITIONAL_OPTS_NIGHTLY}                        \
                           ${DB_COMMON_OPTS}
