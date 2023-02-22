export ENV_NAME=hm
export DATASETS_PWD=/localdisk/benchmark_datasets
# export DB_COMMON_OPTS="-db_server ansatlin07.an.intel.com -db_port 3306 -db_user gashiman -db_pass omniscidb -db_name omniscidb"
# export DB_COMMON_OPTS="-db_name database_test.sqlite"
export DB_COMMON_OPTS="-db_server ansatlin07.an.intel.com -db_port 3306 -db_user gashiman -db_pass omniscidb -db_name omniscidb -db_driver mysql+mysqlconnector"
export ADDITIONAL_OPTS="-iterations 1"
