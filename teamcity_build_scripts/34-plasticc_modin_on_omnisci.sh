#!/bin/bash -e

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
mkdir -p ${PWD}/tmp
python3 run_modin_benchmark.py -bench_name plasticc -data_file '${DATASETS_PWD}/plasticc/'                                          \
                              -pandas_mode Modin_on_omnisci -ray_tmpdir ${PWD}/tmp                                                 \
                              -commit_omnisci ${BUILD_REVISION}                                                                    \
                              -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                              -commit_modin ${BUILD_MODIN_REVISION}                                                                \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS} ${DB_PLASTICC_OPTS}

conda deactivate
