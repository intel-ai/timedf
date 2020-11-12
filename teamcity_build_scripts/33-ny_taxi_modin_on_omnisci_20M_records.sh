#!/bin/bash

source ${CONDA_PREFIX}/bin/activate
conda activate ${ENV_NAME}
mkdir -p ${PWD}/tmp
python3 run_ibis_benchmark.py -bench_name ny_taxi                                                                                  \
                              -data_file '${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv'              \
                              -pandas_mode Modin_on_omnisci -ray_tmpdir ${PWD}/tmp                                                 \
                              -commit_omnisci ${BUILD_REVISION}                                                                    \
                              -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                              -commit_modin ${BUILD_MODIN_REVISION}                                                                \
                              ${ADDITIONAL_OPTS}                                                                                   \
                              ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                              ${DB_COMMON_OPTS} ${DB_H2O_OPTS}

conda deactivate
