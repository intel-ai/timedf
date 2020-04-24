python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --save_env True --python_version 3.7 --task benchmark       \
                          --ci_requirements "${PWD}"/ci_requirements.yml                                                      \
                          --ibis_path "${PWD}"/../ibis/                                                                       \
                          -executable "${PWD}"/../omniscidb/build/bin/omnisci_server                                          \
                          -database_name ${DATABASE_NAME} -table santander -bench_name santander -dfiles_num 1 -iterations 5  \
                          -data_file '/localdisk/benchmark_datasets/santander/train.csv.gz'                                   \
                          -pandas_mode Pandas -ray_tmpdir /tmp -validation True                                               \
                          -commit_omnisci ${BUILD_REVISION} -commit_ibis ${BUILD_IBIS_REVISION}                               \
                          -commit_omniscripts ${BUILD_OMNISCRIPTS_REVISION}                                                   \
                          -db_server ${DATABASE_SERVER_NAME} -db_port 3306                                                    \
                          -db_user ${DATABASE_USER_NAME} -db-pass "${DATABASE_USER_PW}"                                       \
                          -db_name "${DATABASE_NAME}" -db_table_etl santander_etl -db_table_ml santander_ml

