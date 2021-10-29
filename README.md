# Scripts that are used to run modin-specific benchmarks in automated way in TeamCity and for performance analysis in development cycle.

## Requirements

Scripts require miniconda to be installed.

OmnisciDB server which is used in corresponding Modin backend often requires a lot of open files, so it is a good idea to run it with
`ulimit -n 10000`.

## Main benchmarks launching script

Main script is called `run_modin_tests.py`.
Script automatically creates conda environment if it doesn't exist or you want to recreate it.
All subsequent work is being done in created conda environment. Environment can be
removed or saved after executing.
Results can be stored in MySQL database and visualized using Grafana charts.
The main script calls `run_modin_benchmark.py` which is responsble for launch of specified benchmark using provided arguments.

Sample run taxi benchmark command line:
```
python3 run_modin_tests.py --env_name modin-test --env_check True --python_version 3.7 --task benchmark --save_env True --bench_name ny_taxi --iters 5 --ci_requirements ./ci_requirements.yml -data_file '${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv'
```

Sample run census benchmark command line:
```
python3 run_modin_tests.py --env_name modin-test --env_check True --python_version 3.7 --task benchmark -bench_name census -data_file ./census/ipums_education2income_1970-2010.csv.gz -pandas_mode Modin_on_omnisci -ray_tmpdir ./tmp
```

More examples could be find in scripts of `teamcity_build_scripts`. 
Also there is `test_run_script.sh` which can be served as example of all steps that have to be done for benchmarks launching.

## Standalone benchmark launch

TBD
