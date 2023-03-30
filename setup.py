from pathlib import Path
from distutils.core import setup

reqs_dir = Path(__file__).parents[0] / "requirements"


def parse_reqs(name):
    with open(reqs_dir / name, "r") as f:
        return f.readlines()


reporting_reqs = parse_reqs("reporting.txt")
dev_reqs = parse_reqs("linters.txt") + parse_reqs("unittests.txt")

all_reqs = reporting_reqs + dev_reqs

setup(
    name="omniscripts",
    version="0.1",
    description="Tools for benchmarking data processing with data frames",
    author="???",
    author_email="???",
    url="https://github.com/intel-ai/omniscripts/",
    packages=["omniscripts"],
    # I suggest we migrate from requirements.yaml to requirements.txt and just put these file here:
    install_requires=parse_reqs("base.txt"),
    extras_require={"reporting": reporting_reqs, "dev": dev_reqs, "all": all_reqs},
    python_requires=">=3.8",
)
