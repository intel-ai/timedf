python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --python_version 3.7                      \
--task benchmark --ci_requirements "${PWD}"/ci_requirements.yml --save_env True --ibis_path                 \
"${PWD}"/../ibis/ --executable "${PWD}"/../omniscidb/build/bin/omnisci_server                               \
-calcite_port 62225 -http_port 1718 -port 1717 -user admin -password HyperInteractive                       \
-database_name ${DATABASE_NAME} -table plasticc -bench_name plasticc -dfiles_num 1 -iterations 5            \
-data_file '/localdisk/benchmark_datasets/plasticc/'                                                        \
-pandas_mode Pandas -ray_tmpdir /tmp -validation True                                                       \
-db_server ansatlin07.an.intel.com -db_port 3306 -db_user gashiman -db_pass omniscidb -db_name omniscidb    \
-db_table_etl plasticc_etl -db_table_ml plasticc_ml                                                         \
-commit_omnisci ${BUILD_REVISION} -commit_ibis ${BUILD_IBIS_REVISION}