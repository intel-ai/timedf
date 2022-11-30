### Intro
This is a folder with everything necessary to reproduce key benchmarks. 

### Files
- `all.sh` - Main script to run everything. Before running it, change `DATASETS_ROOT` to a location where you want to store datasets (around 40GB).
- `load_data.sh` - script that loads datasets
- `run_docker.sh` - script that starts docker container and automatically start running benchmarks
- `run_benchmarks.sh` - scripts that is copied to the container to run benchmarks.

### Requirements
- docker
- aws cli