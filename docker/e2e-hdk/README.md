### Intro
This is a folder with everything necessary to reproduce key benchmarks. 

### Requirements
- docker - to build and run the container

### How to run 
0. You need to have complete omniscripts repository before running, so clone it or download current version.
1. *Optional.* Open `all.sh` and modify `RESULTS` variable. This is where the benchmark results will be stored. By default they are stored in the current folder.
2. Change dir to this folder `cd omniscripts/docker/e2e-hdk`
3. Run `./all.sh -d DATASETS_PATH [-r xlsx]` to benchmark. Results will be stored in `RESULTS` folder (in the current folder in `results` dir by default). 
    1. `DATASETS_PATH` is where the datasets will be stored, so expect this filder to have 170GB of data. 
    2. Optional `-r xlsx` will generate xlsx report with benchmark results.
### Files
- `all.sh` - Main script to run everything.
- `load_data.sh` - script that loads datasets
- `run_docker.sh` - script that starts docker container and automatically start running benchmarks
- `run_benchmarks.sh` - scripts that is copied to the container to run benchmarks.
