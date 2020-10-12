# Benchmarking       scripts that are used to run OmniSciDB benchmarks in automated way in TeamCity and for performance analyzes in development cycle.

## Requirements
Scripts require to be installed:
* the following python3 packages: pymapd, braceexpand, mysql-connector-python;
* conda or miniconda for ibis tests and benchmarks.

OmnisciDB server often
requires a lot of open files, so it is a good idea to run it with
`ulimit -n 10000`.

## Main benchmark script

Main script is called `run_omnisci_benchmark.py`. It has two distinct
modes of operation: to run synthetic benchmarks and to run dataset
benchmarks. Some command line switches are different in these modes.

Here go switches common to both modes:

Switch | Default value | Meaning
------ | ------------- | -------
-r, --report | report.csv | Name of file to write report into.
-path | | Path to omniscidb/Benchmarks directory.
-e, --executable | | Path to omnisci_server executable.
-w, --workdir | | Path to omnisci working directory. By default parent directory of executable location is used. Data directory is used in this location.
-o, --port | 62274 | TCP port number to run omnisci_server on.
-u, --user | admin | User name to use on omniscidb server.
-p, --passwd | HyperInteractive | User password to use on omniscidb server.
-n, --name | omnisci | Database name to use on omniscidb server.
-t, --import-table-name | | Name of table to import data to. NOTE: This table will be dropped before and after the import test.
-l, --label | | Benchmark run label. Required by omniscidb benchmark scripts.
-m, --mode | | Select benchmark mode. It is either synthetic or dataset.
-fs, --fragment-size | | Fragment size to use for created table. Multiple values are allowed and encouraged. If no -fs switch is specified, default fragment size is used and templated CREATE TABLE sql files cannot be used. Fragment size is required for synthetic tests.

Switches other than `-fs` that are required for _synthetic_ tests:

Switch | Default value | Meaning
------ | ------------- | -------
-nf, --num-fragments | | Number of fragments to generate for synthetic benchmark. Dataset size is fragment_size * num_fragments.
-sq, --synthetic-query | | Synthetic benchmark query group.

Switches required for _dataset_ tests:

Switch | Default value | Meaning
------ | ------------- | -------
-f, --import-file | | Absolute path to file or wildcard on omnisci_server machine with data for import test. If wildcard is used, all files are imported in one COPY statement. Limiting number of files is possible using curly braces wildcard, e.g. trips_xa{a,b,c}.csv.gz.
-c, --table-schema-file | | Path to local file with CREATE TABLE sql statement for the import table.
-d, --queries-dir | | Absolute path to dir with query files.

The following switches if specified, allow recording results in a
MySQL database:

Switch | Default value | Meaning
------ | ------------- | -------
-db-server | localhost | Host name of MySQL server.
-db-port | 3306 | Port number of MySQL server.
-db-user | | Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.
-db-pass | omniscidb | Password to use to connect to MySQL database.
-db-name | omniscidb | MySQL database to use to store benchmark results.
-db-table | | Table to use to store results for this benchmark.
-commit | 1234567890123456789012345678901234567890 | Commit hash to use to record this benchmark results.

Script automatically starts up omniscidb server, creates and
initializes data directory if it doesn't exist or it is not
initialized.

Sample synthetic command line:
```
python3 run_omnisci_benchmark.py -m synthetic -path=/localdisk/username/omniscidb/Benchmarks -u admin -p HyperInteractive -e /localdisk/username/omniscidb/release/bin/omnisci_server -n omnisci -t baseline_hash_benchmark -l baseline_hash_test -nf 10 -sq BaselineHash -i 5 -fs 2000000 -fs 5000000
```

Sample dataset command line with report to MySQL database:
```
python3 run_omnisci_benchmark.py -m dataset -path=/localdisk/username/omniscidb/Benchmarks -u admin -p HyperInteractive -e /localdisk/username/omniscidb/release/bin/omnisci_server -n omnisci -t flights_benchmark -l flights_test -f /localdisk/benchmark_datasets/flights/flights_2008_7M/flights_2008_7M.csv -c /localdisk/username/omniscidb/Benchmarks/import_table_schemas/flights_56_columns.sql -d /localdisk/username/omniscidb/Benchmarks/queries/flights -i 5 -fs 2000000 -fs 5000000 -db-server=mysqlserver -db-user=mysqluser -db-pass=omniscidb -db-name=omniscidb -db-table=flightsbench
```

## Taxi pandas script

Pandas script name is `taxi/taxibench_pandas.py`. Pandas is required
to run this script.

Following are switches that are required to run it:

Switch | Default value | Meaning
------ | ------------- | -------
-r | report_pandas.csv | Report file name.
-df | 1 | Number of datafiles to input into database for processing.
-dp | | Wildcard pattern of datafiles that should be loaded.
-i | 5 | Number of iterations to run every benchmark. Best result is selected.

Database reporting switches are the same as for main benchmark script.

Sample script command line:
```
python3 taxi/taxibench_pandas.py -df 2 -i 5 -dp '/datadir/taxi/trips_*.csv.gz'
```

## Ibis script

Ibis build, tests and benchmarks run through `run_ibis_test.py`. It has three distinct
modes of operation: 
* build and install ibis;
* run ibis tests using pytest;
* run benchmarks using Omnisci.

Parameters which can be used:

Switch | Default value | Meaning
------ | ------------- | -------
-t, --task | | Task for execute from supported list [build, test, benchmark]. Use "," separator for multiple tasks. 
-en, --env_name | ibis-tests | Conda env name.
-ec, --env_check | False | Check if env exists. If it exists don't recreate.
-s, --save_env | False | Save conda env after executing.
-r, --report_path | parent dir of omnscripts | Path to report file.
-ci, --ci_requirements | ci_requirements.yml | File with ci requirements for conda env.
-py, --python_version | 3.7 | File with ci requirements for conda env.
-i, --ibis_path | | Path to ibis directory.
-e, --executable | | Path to omnisci_server executable.
-w, --workdir | | Path to omnisci working directory. By default parent directory of executable location is used. Data directory is used in this location.
-o, --omnisci_port | 6274 | TCP port number to run omnisci_server on.
-u, --user | admin | User name to use on omniscidb server.
-p, --password | HyperInteractive | User password to use on omniscidb server.
-n, --name | agent_test_ibis | Database name to use in omniscidb server.
-commit_omnisci | 123456... | Omnisci commit hash to use for tests.
-commit_ibis | 123456... | Ibis commit hash to use for tests.

For ibis tests:

Switch | Default value | Meaning
------ | ------------- | -------
--expression | " " | Run tests which match the given substring test names and their parent classes. Example: 'test_other', while 'not test_method' matches those that don't contain 'test_method' in their names.
For benchmark task and recording its results in a MySQL database:

Switch | Default value | Meaning
------ | ------------- | -------
-bn, --bench_name | | Benchmark name from supported list [ny_taxi, santander]
-db-server | localhost | Host name of MySQL server.
-db-port | 3306 | Port number of MySQL server.
-db-user | | Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.
-db-pass | omniscidb | Password to use to connect to MySQL database.
-db-name | omniscidb | MySQL database to use to store benchmark results.
-db-table | | Table to use to store results for this benchmark.
-df, --dfiles_num | 1 | Number of datafiles to input into database for processing.
-dp, --dpattern |  | Wildcard pattern of datafiles that should be loaded.
-it, --iters | 5 | Number of iterations to run every query. Best result is selected.

Script automatically creates conda environment if it doesn't exist or you want to recreate it,
starts up omniscidb server, creates and initializes data directory if it doesn't exist or it is not
initialized. All subsequent work is being done in created conda environment. Environment can be
removed or saved after executing.

Sample build ibis command line:
```
python3 run_ibis_tests.py --env_name ibis-test --env_check False --save_env True --python_version 3.7 --task build --name agent_test_ibis --ci_requirements /localdisk/username/omniscripts/ci_requirements.yml --ibis_path /localdisk/username/ibis/ --executable /localdisk/username/omniscidb/release/bin/omnisci_server
```

Sample run ibis tests command line:
```
python3 run_ibis_tests.py --env_name ibis-test --env_check True --save_env True --python_version 3.7 --task test --name agent_test_ibis --report /localdisk/username/ --ibis_path /localdisk/username/ibis/ --executable /localdisk/username/omniscidb/build/bin/omnisci_server --user admin --password HyperInteractive
```

Sample run taxi benchmark command line:
```
python3 run_ibis_tests.py --env_name ibis-test --env_check True --python_version 3.7 --task benchmark --ci_requirements /localdisk/username/omniscripts/ci_requirements.yml --save_env True --report /localdisk/username/ --ibis_path /localdisk/username/ibis/ --executable /localdisk/username/omniscidb/build/bin/omnisci_server -u admin -p HyperInteractive -n agent_test_ibis --bench_name ny_taxi --dfiles_num 20 --dpattern '/localdisk/username/benchmark_datasets/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv.gz' --iters 5 -db-server localhost -db-port 3306 -db-user user -db-pass omniscidb -db-name omniscidb -db-table taxibench_ibis
```

Sample run santander benchmark command line:
```
python3 run_ibis_tests.py --env_name ibis-test --env_check True --python_version 3.7 --task benchmark --ci_requirements /localdisk/username/omniscripts/ci_requirements.yml --save_env True --report /localdisk/username/ --ibis_path /localdisk/username/ibis/ --executable /localdisk/username/omniscidb/build/bin/omnisci_server -u admin -p HyperInteractive -n agent_test_ibis --bench_name santander --dpattern '/localdisk/benchmark_datasets/santander/train.csv.gz' --iters 5 -db-server localhost -db-port 3306 -db-user user -db-pass omniscidb -db-name omniscidb -db-table santander_ibis
```
