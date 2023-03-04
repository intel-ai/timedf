#!/bin/bash -e

set -eux

export CONDA_PREFIX=~/miniconda3
export ENV_NAME=hdk_test

eval echo Removing ${CONDA_PREFIX} ...
eval rm -rf ${CONDA_PREFIX} 

echo Miniconda installation ...
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda3.sh
eval bash /tmp/miniconda3.sh -b -p ${CONDA_PREFIX} -f -u
rm -f /tmp/miniconda3.sh
# The following expands tilde in ${CONDA_PREFIX} and runs command
eval ${CONDA_PREFIX}/bin/conda init bash

echo Miniconda activation ...
eval source ${CONDA_PREFIX}/bin/activate
conda update -n base -c defaults conda -y

eval source ${CONDA_PREFIX}/bin/activate

#export OMNISCIDB_BUILD_ENV=omnisci-build

conda env remove --name ${ENV_NAME}  -y
conda create --name ${ENV_NAME}  python=3.8 -y
conda env update --name ${ENV_NAME} -f /hdk/omniscidb/scripts/mapd-deps-conda-dev-env.yml

conda activate ${ENV_NAME}

conda install -c conda-forge cmake==3.23.1

echo ==== conda list of ${ENV_NAME} start
conda list
echo ==== conda list of ${ENV_NAME} end

if [ -d "build" ]; then
	rm -Rf "build"
fi
mkdir build; cd build
cmake  .. -DENABLE_CUDA=off -DCMAKE_BUILD_TYPE=release
make -j`nproc`
make install


#eval source ${CONDA_PREFIX}/bin/activate

# remove previously installed packages
# conda clean -f
#./teamcity_build_scripts/19-build_modin_dbe.sh


eval source ${CONDA_PREFIX}/bin/activate ${ENV_NAME}

# we have already installed omniscidb; just comment the line
# sed -i '/- pyomniscidbe/s/^/#/' modin/requirements/env_omnisci.yml
# conda env update --name ${ENV_NAME} --file modin/requirements/env_omnisci.yml
conda install psutil braceexpand scikit-learn==1.0.2 xgboost scikit-learn-intelex mysql mysql-connector-python sqlalchemy>=1.4 -c conda-forge

# Modin installation
cd modin && pip install -e . && pip install .[ray] ray==2.0.1 && cd ..

echo ==== conda list of ${ENV_NAME} start
conda list
echo ==== conda list of ${ENV_NAME} end

python -c "import pyhdk"

# TEMP step while we have hdk support in separate modin branch not in master! 
#conda env remove --name ${ENV_NAME}_tmp  -y
#conda create --name ${ENV_NAME}_tmp  python=3.8 -y
#conda activate ${ENV_NAME}_tmp
#git clone https://github.com/modin-project/modin modin_master
#conda install psutil braceexpand scikit-learn==1.0.2 xgboost scikit-learn-intelex mysql mysql-connector-python -c conda-forge

#cd modin_master && pip install -e . && pip install .[ray]


#!/bin/bash -e

#eval source ${CONDA_PREFIX}/bin/activate

# remove previously installed packages
# conda clean -f
#./teamcity_build_scripts/19-build_modin_dbe.sh


eval source ${CONDA_PREFIX}/bin/activate ${ENV_NAME}

# we have already installed omniscidb; just comment the line
# sed -i '/- pyomniscidbe/s/^/#/' modin/requirements/env_omnisci.yml
# conda env update --name ${ENV_NAME} --file modin/requirements/env_omnisci.yml
conda install psutil braceexpand scikit-learn==1.0.2 xgboost scikit-learn-intelex mysql mysql-connector-python sqlalchemy>=1.4 -c conda-forge

# Modin installation
cd modin && pip install -e . && pip install .[ray] ray==2.0.1 && cd ..

echo ==== conda list of ${ENV_NAME} start
conda list
echo ==== conda list of ${ENV_NAME} end

python -c "import pyhdk"

# TEMP step while we have hdk support in separate modin branch not in master! 
#conda env remove --name ${ENV_NAME}_tmp  -y
#conda create --name ${ENV_NAME}_tmp  python=3.8 -y
#conda activate ${ENV_NAME}_tmp
#git clone https://github.com/modin-project/modin modin_master
#conda install psutil braceexpand scikit-learn==1.0.2 xgboost scikit-learn-intelex mysql mysql-connector-python -c conda-forge

#cd modin_master && pip install -e . && pip install .[ray]

# The `LD_PRELOAD` variable can be viewed in the next build step via `export` command.
# expand '~'
eval LD_PRELOAD=${CONDA_PREFIX}/envs/${ENV_NAME}/lib/libtbbmalloc_proxy.so.2
eval LD_LIBRARY_PATH=${CONDA_PREFIX}/envs/${ENV_NAME}/lib/

echo "##teamcity[setParameter name='env.LD_PRELOAD' value='${LD_PRELOAD}']"
echo "##teamcity[setParameter name='env.LD_LIBRARY_PATH' value='${LD_LIBRARY_PATH}']"

eval source ${CONDA_PREFIX}/bin/activate

PANDAS_MODE="Modin_on_hdk" ./build_scripts/ny_taxi_ml.sh

eval source ${CONDA_PREFIX}/bin/activate

PANDAS_MODE="Modin_on_ray" ./build_scripts/ny_taxi_ml.sh


eval source ${CONDA_PREFIX}/bin/activate

PANDAS_MODE="Pandas" ./build_scripts/ny_taxi_ml.sh
