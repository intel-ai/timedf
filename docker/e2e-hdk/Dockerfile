FROM ubuntu:20.04

ARG http_proxy=http://proxy-dmz.intel.com:912
ARG https_proxy=http://proxy-dmz.intel.com:912

ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

RUN apt-get update --yes \
    && apt-get install wget git --yes && \
    rm -rf /var/lib/apt/lists/*

ENV USER=modin
ENV UID=1000
ENV HOME=/home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --home $HOME \
    $USER

RUN mkdir /datasets && mkdir /results
RUN chown -R ${USER}:${USER} /datasets /results

WORKDIR ${HOME}

ADD omniscripts.tar omniscripts
RUN chown -R ${USER}:${USER} omniscripts

# Done with actions that require root, switching to user mode

USER ${USER}
SHELL ["/bin/bash", "--login", "-c"]

ENV CONDA_PREFIX=${HOME}/miniconda

ENV ENV_NAME=modin_hdk



RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_PREFIX}" -f -u && \
    "${CONDA_PREFIX}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh && \
    echo ". '${CONDA_PREFIX}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

# conda env setup
RUN conda update -n base -c defaults conda -y && \
    conda create --name modin_hdk && \
    conda activate modin_hdk && \
    conda install python=3.9 -c conda-forge -y && \
    conda env update -n modin_on_hdk --file=omniscripts/ci_requirements.yml && \
    conda install mamba -c conda-forge -y && \
    mamba install modin-hdk -c conda-forge && \
    conda clean --all --yes 

ENV DATASETS_PWD /datasets
ENV TBB_MALLOC_USE_HUGE_PAGES 1


COPY run_benchmarks.sh ${HOME}

ENTRYPOINT ["bash", "run_benchmarks.sh"]
