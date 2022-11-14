# Scripts that are used to run modin-specific benchmarks in automated way in TeamCity and for performance analysis in development cycle.

## Requirements

Scripts require miniconda to be installed.

OmnisciDB server which is used in corresponding Modin backend often requires a lot of open files, so it is a good idea to run it with
`ulimit -n 10000`.

## Main benchmarks launching script

Main script is called `run_modin_tests.py`.
Script automatically creates conda environment if it doesn't exist or you want to recreate it (```--env_check``` flag controls this - ```False``` means env recreation).
All subsequent work is being done in created conda environment. Environment can be removed or saved after executing (```--save_env``` flag is responsbile for this - ```False``` means removal).
Results can be stored in MySQL database and visualized using Grafana charts.

Sample run taxi benchmark command line (ci_requirements.yml should contain description for ```modin-test``` env):
```
python3 run_modin_tests.py --env_name modin-test --env_check True -task benchmark --save_env True --bench_name ny_taxi --iters 5 --ci_requirements ./ci_requirements.yml -data_file '${DATASETS_PWD}/taxi/trips_xa{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t}.csv'
```

Sample run census benchmark command line (note that ```modin-test``` env will be deleted after test run finishes):
```
python3 run_modin_tests.py --env_name modin-test --env_check True -task benchmark -bench_name census -data_file ./census/ipums_education2income_1970-2010.csv.gz -pandas_mode Modin_on_omnisci -ray_tmpdir ./tmp
```

More examples could be find in scripts of `teamcity_build_scripts`. Those scripts contain actual examples for running each benchmark we have now.
Also there is `test_run_script.sh` which can be served as example of all steps that have to be done for benchmarks launching.

## Standalone benchmark launch

TBD
