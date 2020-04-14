python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --python_version 3.7                      \
--task benchmark --ci_requirements "${PWD}"/ci_requirements.yml --save_env True --ibis_path                 \
"${PWD}"/../ibis/ --executable "${PWD}"/../omniscidb/build/bin/omnisci_server                               \
-calcite_port 62225 -http_port 1718 -port 1717 -user admin -password HyperInteractive                       \
-database_name ${DATABASE_NAME} -table census -bench_name census -dfiles_num 1 -iterations 5                \
-fragments_size 1000000                                                                                     \
-data_file '/localdisk/benchmark_datasets/census/ipums_education2income_1970-2010.csv.gz'                   \
-pandas_mode Pandas -ray_tmpdir /tmp -validation True -optimizer stock                                      \
-db_server ansatlin07.an.intel.com -db_port 3306 -db_user gashiman -db_pass omniscidb -db_name omniscidb    \
-db_table_etl census_etl -db_table_ml census_ml                                                             \
-commit_omnisci ${BUILD_REVISION} -commit_ibis ${BUILD_IBIS_REVISION}