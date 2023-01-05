#!/bin/bash -e

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
# This variable is used to improve performance and its value was obtained during the experiment.
# Each workload can have a different value.
export MODIN_HDK_FRAGMENT_SIZE=32000000
mkdir -p ${PWD}/tmp
python3 run_modin_tests.py -bench_name plasticc -data_file "${DATASETS_PWD}/plasticc/"                                             \
                              -task benchmark -pandas_mode Modin_on_hdk                                                            \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS}

conda deactivate
