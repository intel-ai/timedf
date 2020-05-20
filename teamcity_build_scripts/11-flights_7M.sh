ROOT_DIR="${PWD}"
cd omniscripts
python3 run_omnisci_benchmark.py --env_name ${ENV_NAME} --env_check True --save_env True --mode dataset                             \
                                 --ci_requirements ci_requirements.yml                                                              \
                                 --executable "${ROOT_DIR}"/omniscidb/build/bin/omnisci_server                                      \
                                 --import-table-name flights_benchmark --label flights_test                                         \
                                 --table-schema-file "${ROOT_DIR}"/omniscidb/Benchmarks/import_table_schemas/flights_56_columns.sql \
                                 --port 61274 --name omnisci --user admin --passwd HyperInteractive                                 \
                                 -path "${ROOT_DIR}"/omniscidb/Benchmarks                                                           \
                                 --import-file "/localdisk/benchmark_datasets/flights/flights_2008_7M/flights_2008_7M.csv"          \
                                 --queries-dir "${ROOT_DIR}"/omniscidb/Benchmarks/queries/flights                                   \
                                 --iterations 5 --fragment-size 5000000                                                             \
                                 -commit ${BUILD_REVISION}                                                                          \
                                 -db-server ${DATABASE_SERVER_NAME} -db-name "${DATABASE_NAME}" -db-table flightsbench              \
                                 -db-user ${DATABASE_USER_NAME} -db-pass "${DATABASE_USER_PW}" 
