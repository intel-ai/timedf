#!/bin/bash -e

if [ -z ${ADDITIONAL_OPTS+x} ]; then
    export "ADDITIONAL_OPTS=-dfiles_num 10 -no_pandas True"
else
    # ny_taxy doesn`t have ML. Because of it, it give warning if any value for option -no_ml specifyed
    export "ADDITIONAL_OPTS=${ADDITIONAL_OPTS/-no_ml True/} -dfiles_num 1"
fi

# At this moment NYC taxi benchmark works fine with import_mode copy-from flag and fails during import by fsi
# on the trips_xad.csv with message: Exception: Invalid: In CSV column #1:
#    CSV conversion error to int64: invalid value 'VTS'

python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 -task benchmark         \
                          --ci_requirements "${PWD}/ci_requirements.yml"                                                       \
                          --ibis_path "${PWD}/../ibis/"                                                                        \
                          -executable "${PWD}/../omniscidb/build/bin/omnisci_server"                                           \
                          -user admin -password HyperInteractive                                                               \
                          -database_name ${DATABASE_NAME} -table taxi -bench_name ny_taxi -iterations 1                        \
                          -data_file '${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv.gz'           \
                          -pandas_mode Pandas -ray_tmpdir /tmp                                                                 \
                          -commit_omnisci ${BUILD_REVISION}                                                                    \
                          -commit_ibis ${BUILD_IBIS_REVISION}                                                                  \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                    \
                          ${ADDITIONAL_OPTS}                                                                                   \
                          ${DB_COMMON_OPTS} ${DB_TAXI_OPTS}
