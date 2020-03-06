python3 run_ibis_tests.py --env_name ${ENV_NAME} --env_check True --python_version 3.7                         \
--task benchmark --ci_requirements "${PWD}"/ci_requirements.yml --save_env True                                \
--report "${PWD}"/.. --ibis_path "${PWD}"/../ibis/ --executable "${PWD}"/../omniscidb/build/bin/omnisci_server \
-u admin -p HyperInteractive -n ${DATABASE_NAME} --bench_name plasticc                                         \
--dpattern '/localdisk/benchmark_datasets/plasticc'                                                            \
-db-server ansatlin07.an.intel.com -db-port 3306 -db-pass omniscidb -db-name omniscidb -db-table plasticc      \
-commit_omnisci ${BUILD_REVISION} -commit_ibis ${BUILD_IBIS_REVISION}