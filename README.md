# LLaCIE

[![PyPI - Version](https://img.shields.io/pypi/v/llacie)](https://pypi.org/project/llacie/) [![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/llacie.svg)](https://pypi.org/project/llacie/) [![CI](https://img.shields.io/github/actions/workflow/status/tpaklab/llacie/ci.yml?branch=main&label=build&logo=GitHub)](https://github.com/tpaklab/llacie/actions/workflows/ci.yml)

**Large Language (model) Clinical Information Extractor**

This is an [information extraction](http://en.wikipedia.org/wiki/Information_extraction) pipeline that specializes in running [large language models](https://en.wikipedia.org/wiki/Large_language_model) across many clinical notes to abstract new variables.

The task implemented in this initial release is the extraction of presenting signs and symptoms in admission notes for patients with possible infection. This is further detailed in our publication:

- Pak TR, Kanjilal S, McKenna CS, Hoffner-Heinike A, Rhee C, Klompas M. Syndromic Analysis of Sepsis Cohorts Using Large Language Models. _JAMA Netw Open._ 2025 Oct 1;8(10):e2539267. doi:[10.1001/jamanetworkopen.2025.39267](https://doi.org/10.1001/jamanetworkopen.2025.39267). PMID: 41134571; PMCID: PMC12552932.

The pipeline is designed to be extensible to many tasks. It also allows for the comparison of multiple strategies for each task by evaluating each strategy's performance against a gold standard, e.g., a human-labeled dataset.

## Quickstart and demo

Docker is the quickest way to start using this package, because all dependencies (like a Postgres database) can be managed within a single container. If you are new to it, [Docker Desktop](https://www.docker.com/products/docker-desktop/) is likely the easiest way to install Docker. Your Docker environment will need [at least 8GB of RAM](https://docs.docker.com/desktop/settings-and-maintenance/settings/#resources).

Clone this repo, `cd` into it, and run the following. This will take several minutes to build and run the container:

```bash
$ docker-compose up -d
$ docker-compose exec llacie bash
```

If this worked, you should now be in a shell within the container with access to the `llacie` CLI. Run this command to see the main menu, which outlines the basic steps of the pipeline.

```bash
$ llacie
```

To automatically download the Llama model files from HuggingFace, you need to [request access to the Llama 3 8B model](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [create an access token](https://huggingface.co/settings/tokens) for yourself, and save it into the container.

```bash
$ hf auth whoami
$ hf auth login   # If the prior command says, "Not logged in".
                  # If asked to "Add token as git credential?", answer no.
```

We can now run the example analysis on 100 synthetic admission notes, of which 20 have "gold standard" human-created labels for presenting signs/symptoms. For simplicity, the example uses a quantized version of Llama 3 8B that fits in ~6GB of RAM and runs on CPU only.

```bash
$ llacie init-db
$ llacie import-notes text examples/admission-100.txt
$ llacie sections extract -s regex
$ llacie features extract -s llama3_8b
$ llacie episode-labels extract -s pres_sx_eplab2.llama3_8b
$ llacie episode-labels import pres_sx_eplab2 examples/admission-100-labels.xlsx
$ llacie episode-labels evaluate
```

## Installing from PyPI

You can install the package directly [from PyPI](https://pypi.org/project/llacie/), which requires Python ≥3.11.

```bash
$ pip install llacie
```

Although this will install some of the Python package dependencies, note that you will need to set up a Postgres database and configure `llacie` to connect to it. 

### Configuration

Copy `.env.example` to `.env`, and edit the variables within.

The base package runs LLMs using [llama-cpp-python](https://pypi.org/project/llama-cpp-python/) on CPU only, but for faster inference, you'll likely want to [install vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/). We don't do this by default because vLLM installation has to be customized to your specific hardware and CUDA version (for NVIDIA GPUs).

## Installing a development environment

### Using conda

Create or activate a conda environment that includes Python 3.11 and the `psycopg2` package, e.g.

```bash
$ conda create -n llacie python=3.11 psycopg2  # First time only
$ conda activate llacie                        # Subsequent times
(llacie) $
```

We develop on this package in a `venv` (aka virtualenv) within this repository, as this allows the package to be installed in `--editable` mode, so we can work on it and use it simultaneously.

```bash
(llacie) $ python3 -m venv .venv
(llacie) $ . .venv/bin/activate
```

If that worked, the shell prompt is now also prefixed with `(.venv)`. We next install the repo itself as a local module in this virtualenv. This will also automatically download and install dependencies enumerated in `pyproject.toml`.

> **Important**: Installing dependencies requires a C/C++ compiler. If this step fails on the [MGB Linux cluster](https://rc.partners.org/kb/computational-resources/faq-eristwo?article=1553), run `module load gcc/9.3.0` and try again.

```bash
(.venv) (llacie) $ pip install -e .[dev]
```

If everything worked, you should be able to see the main menu by running:

```bash
(.venv) (llacie) $ llacie
```

### Running tests

The test suite is in `tests/`. Currently, this runs integration tests based on the Quickstart demo, checking the command outputs and that database state is updated appropriately after each step. Common test suite invocations can be run with `make`:

```bash
make test-install
make test           # Runs all of the tests
make test-fast      # Runs only the quicker tests that don't require LLM inference
```

We automatically run the test suite for every commit pushed to this repo using [Github Actions](https://github.com/tpaklab/llacie/actions/workflows/ci.yml).

### Building the package

The package is Python-only and can be built using [flit](https://flit.pypa.io/en/stable/).

```bash
$ flit build
$ flit publish
```

## Citation 

If you use LLaCIE for your research, please cite our publication:

- Pak TR, Kanjilal S, McKenna CS, Hoffner-Heinike A, Rhee C, Klompas M. Syndromic Analysis of Sepsis Cohorts Using Large Language Models. _JAMA Netw Open._ 2025 Oct 1;8(10):e2539267. doi:[10.1001/jamanetworkopen.2025.39267](https://doi.org/10.1001/jamanetworkopen.2025.39267). PMID: 41134571; PMCID: PMC12552932.