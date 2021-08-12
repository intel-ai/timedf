## Benchmarking scripts that are used to run Flytekit benchmarks for performance analyzes in development cycle.

## There are implemented following benchmarks:
* taxi
* census
* plasticc
* santander

## Requirements
Scripts require to be installed:
* <a href="https://git-scm.com/">`Git`</a> and <a href="https://www.python.org/downloads/">`python >= 3.7`</a> , `pip3`
* `conda` or `miniconda` for flytekit tests and benchmarks;
* the following python3 packages: `flytekit>=0.20.1` .

## Flytekit installation (recommended using a virtual environment)
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>(venv)$ pip3 install flytekit --upgrade
</pre></div>

## Running benchmarks instructions
* copy and open **jupyter/** `.ipynb` scripts via `jupyter notebook`
* or open  **scrips/** `.py` in text redactor
* replace default path in the input variables in the function signature decorated as `@workflow`
* run sequentially notebook cells or run benchmarks scripts as `(venv)$ python <benchmark_file_name>`

## Flyte app building
As suggested on the <a href="https://docs.flyte.org/en/latest/getting_started.html">Flyte Getting Started page</a>, Flyte project should have the following structure:
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>.
├── Dockerfile
├── docker_build_and_tag.sh
├── myapp
│         ├── __init__.py
│         └── workflows
│             ├── __init__.py
│             └── example.py
└── requirements.txt
</pre></div>

## There is a Flyte app example available - <a href="https://github.com/flyteorg/flytekit-python-template">flytekit-python-template</a>:
* You should clone this repo:

<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>(venv)$ git clone https://github.com/flyteorg/flytekit-python-template.git myflyteapp
(venv)$ cd myflyteapp
</pre></div>

You should run workflows locally as a Python script:
<div class="highlight-text notranslate"><div class="highlight"><pre><span></span>(venv)$ python myapp/workflows/example.py
</pre></div>
  
You may also copy benchmark scripts into `myflyteapp/myapp/workflows/` to build flyte project further
