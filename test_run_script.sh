#!/bin/bash

mkdir test_launch
cd test_launch

# cloning repos
git clone https://github.com/modin-project/modin.git -b master
git clone https://github.com/intel-ai/omniscripts.git -b master

# setup miniconda
mkdir ./miniconda_install_script
wget  -P ./miniconda_install_script "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash ./miniconda_install_script/Miniconda3-latest-Linux-x86_64.sh -p ./miniconda3 -b
source ./miniconda3/bin/activate

# get taxi reduced dataset
mkdir benchmark_datasets
mkdir benchmark_datasets/taxi
wget https://modin-datasets.s3.us-west-2.amazonaws.com/cloud/taxi/trips_xaa.csv --directory-prefix ./benchmark_datasets/taxi/

# create env and run taxi in Modin_on_ray mode
cd omniscripts

export ENV_NAME=omniscripts_test
 
export DATASETS_PWD="../benchmark_datasets"

./teamcity_build_scripts/19-build_modin_dbe.sh
./teamcity_build_scripts/32-ny_taxi_modin_on_ray_20M_records.sh