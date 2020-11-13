#!/bin/bash

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
mkdir -p ${PWD}/tmp
python3 run_ibis_benchmark.py -bench_name census -data_file '${DATASETS_PWD}/census/ipums_education2income_1970-2010.csv.gz'       \
                              -pandas_mode Modin_on_omnisci -ray_tmpdir ${PWD}/tmp                                                 \
                              -commit_omnisci ${BUILD_REVISION}                                                                    \
                              -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                              -commit_modin ${BUILD_MODIN_REVISION}                                                                \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS} ${DB_CENSUS_OPTS}

conda deactivate
