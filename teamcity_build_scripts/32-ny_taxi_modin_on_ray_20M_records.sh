#!/bin/bash -e

mkdir -p ${PWD}/tmp
python3 run_modin_tests.py --env_name ${ENV_NAME} --env_check True --save_env True -task benchmark                              \
                          --ci_requirements "${PWD}/ci_requirements.yml" -bench_name ny_taxi                                   \
                          -data_file '${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv'              \
                          -dfiles_num 1 -pandas_mode Modin_on_ray -ray_tmpdir ${PWD}/tmp                                       \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          -commit_modin ${BUILD_MODIN_REVISION}                                                                \
                          ${ADDITIONAL_OPTS_NIGHTLY}                                                                           \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${DB_COMMON_OPTS} ${DB_TAXI_OPTS}
