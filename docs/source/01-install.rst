Installation
============

To install library with all features::

    git clone https://github.com/intel-ai/timedf.git && cd timedf && pip install ".[all]"


.. _installation-label:

We recommend performing this installation into clean conda environment with ``python=3.9`` that you can create with this command::

    conda create -n timedf python=3.9 -y && conda activate timedf

So the combined command is::

    conda create -n timedf python=3.9 -y && conda activate timedf && \
    git clone https://github.com/intel-ai/timedf.git && cd timedf && pip install ".[all]"
