Some notes for teamcity processes:
1. For results reporting user should have permissions to connect to MySQL host and insert
records there. Modify test scripts with user credentials and MySQL
server host name.
2. To run test scripts user should have installed conda or miniconda.
We are creating conda environment (specified by --env_name flag) from only ci_requirements.yml file
by running run_modin_test.py script with `-task build` flag and then run benchmarks by run_modin_test.py
script with `-task benchmark` flag. All runs are being done in created conda env.