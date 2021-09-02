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

# Proxy settings
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

# Modin settings
ENV RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# Cnvrg.io settings
RUN mkdir "/cnvrg"
WORKDIR "/cnvrg"

# Conda settings
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV_NAME=modin-ray
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN apt-get update --yes && \
    apt-get install wget less git --yes && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u && \
    "${CONDA_DIR}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh

RUN conda update -n base -c defaults conda -y && \
    conda create -n ${CONDA_ENV_NAME} --yes -c conda-forge ${CONDA_ENV_NAME} ray-dashboard scikit-learn scikit-learn-intelex xgboost pip jupyter && \
    conda clean --all --yes

# Activate ${CONDA_ENV_NAME} for interative shells
RUN echo "source /opt/conda/bin/activate ${CONDA_ENV_NAME}" >> /root/.bashrc
# Activate ${CONDA_ENV_NAME} for non-interative shells
RUN sed -e 's,\(\[ -z "\$PS1" \] && return\),# \1,' -i /root/.bashrc
ENV BASH_ENV=/root/.bashrc

# Clean up proxy settings to publish on Docker Hub
ENV http_proxy=
ENV https_proxy=

# Set entrypoint with arguments expansion
ENTRYPOINT ["/bin/bash", "-c", "exec $0 $*"]
