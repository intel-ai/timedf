# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Build image from this dockerfile like this:
# docker build -t modin-on-omnisci:latest --build-arg https_proxy --build-arg http_proxy .


FROM ubuntu:20.04
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV MODIN_BACKEND "omnisci"
ENV MODIN_EXPERIMENTAL "true"

RUN apt-get update --yes \
    && apt-get install wget less git --yes && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir "/cnvrg"
WORKDIR "/cnvrg"
ENV CONDA_DIR /miniconda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u && \
    "${CONDA_DIR}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh && \
    echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

RUN conda update -n base -c defaults conda -y && \
    conda config --set channel_priority strict && \
    conda install -c conda-forge python==3.7.10 && \
    conda install -c intel/label/modin -c conda-forge -c intel modin "ray>=1.0.0" "numpy==1.20.3" "daal4py>=2021.1" && \
    pip install -U jupyter "scikit-learn<0.24.0" xgboost && \
    conda clean --all --yes

ENV http_proxy=
ENV https_proxy=
ENV RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

