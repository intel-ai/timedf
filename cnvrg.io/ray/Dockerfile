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
# docker build -t modin-on-ray:latest --build-arg https_proxy --build-arg http_proxy .

FROM ubuntu:20.04

ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

RUN apt-get update --yes && \
    apt-get install wget less git --yes && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir "/cnvrg"
WORKDIR "/cnvrg"
ENV CONDA_DIR /miniconda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget --quiet --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u && \
    "${CONDA_DIR}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh && \
    echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

RUN conda update -n base -c defaults conda -y && \
    conda install -c conda-forge modin "ray-core>=1.0.0" "ray-dashboard>=1.0.0" "numpy>=1.16.5" scikit-learn "xgboost>=1.3" pip jupyter && \
    conda clean --all --yes

ENV http_proxy=
ENV https_proxy=
ENV RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
