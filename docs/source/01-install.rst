Installation
============

To install library with all features::

    git clone https://github.com/intel-ai/omniscripts.git && cd omniscripts && pip install ".[all]"


We recommend performing this installation into clean conda environment with ``python=3.9`` that you can create with this command::

    conda create -n omniscripts python=3.9 pip -y && conda activate omniscripts

So the combined command is::

    conda create -n omniscripts python=3.9 -y && conda activate omniscripts && \
    git clone https://github.com/intel-ai/omniscripts.git && cd omniscripts && pip install ".[all]"

