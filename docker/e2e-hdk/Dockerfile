FROM ubuntu:20.04

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

WORKDIR ${HOME}

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
    conda install -y python=3.9 mamba sqlalchemy==1.4 psutil braceexpand scikit-learn==1.0.2 xgboost scikit-learn-intelex -c conda-forge && \
    conda activate modin_hdk && mamba install -y modin-hdk -c conda-forge && \
    conda clean --all --yes && \
    pip install --no-cache-dir XlsxWriter

ENV DATASETS_PWD /datasets
ENV DB_COMMON_OPTS="-db_name /results/result_database.sqlite"
ENV TBB_MALLOC_USE_HUGE_PAGES 1

ADD omniscripts.tar omniscripts
# RUN chown -R ${USER}:${USER} omniscripts
# RUN chmod -R 0777 omniscripts

COPY run_benchmarks.sh ${HOME}

CMD ["bash", "run_benchmarks.sh"]
