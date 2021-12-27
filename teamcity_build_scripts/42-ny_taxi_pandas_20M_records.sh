#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                             \
                          -bench_name ny_taxi                                                                                  \
                          -data_file "${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv"              \
                          -dfiles_num 1 -pandas_mode Pandas -ray_tmpdir ${PWD}/tmp                                             \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${DB_COMMON_OPTS} ${DB_TAXI_OPTS}
