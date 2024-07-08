<!-- omit in toc -->
# Modelling Transcriptional Regulation using Gaussian Processes

**Author:** William Purvis

<!-- omit in toc -->
## Table of Contents
- [Project description](#project-description)
- [How to install \& run the project](#how-to-install--run-the-project)
  - [Installing and running in Docker](#installing-and-running-in-docker)
  - [Running locally](#running-locally)
- [How to use the project](#how-to-use-the-project)
  - [Documentation](#documentation)
- [License](#license)

## Project description

This project implements a custom *latent force model* in `GPJax`, a didactic Gaussian Process library for python, to infer the latent activity profile of the p53 transcription factor given the expressions of its target genes. The model is flexible, allowing for different data replicas to be used as well as enabling ablation studies.

## How to install & run the project

The code for this project can either be installed and run within a docker container, or the repository can be run locally. To clone the repository run the following command:

```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/projects/wp289.git
```

### Installing and running in Docker

A dockerfile is provided to run the project alongside all of it's dependencies (defined in environment.yml) in a docker container. To build the docker image from the provided dockerfile run the following commands after cloning the repository:

```bash
docker build -t lfm_image . # builds docker image
```

As the project requires command-line interactions, the docker container needs to be run in interactive mode:

```bash
docker run -it lfm_image
```

The project can then be run within the docker container.

### Running locally

To run this project without docker, create a `conda` environment from the provided `environment.yml` file and activate it:

```bash
conda env create -f environment.yml # create env
conda activate gpjax_wp289          # activate env
```

## How to use the project

To run the `GPJax` implementation of the latent force model (LFM) used to obtain the latent activity of p53 presented in the report run the following command from the root directory:

```bash
python src/main.py
```

In addition, refactored code from the [ALFI](https://github.com/mossjacob/alfi) package is included to validate the results. This LFM can be run using:

```bash
python src/gpytorch/main_alfi.py
```

Additionally, example notebooks are provided as tutorials on how to use this package (`src/notebook.py`).

**NOTE** there is a depency issue with GPJax, and the following warning may appear (that can be safely ignored):

```bash
UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
```

### Documentation

This project uses NumPy-style doc strings and in-depth documentation on how to use the package can be obtained by navigating to the `docs/` directory and building the documentation locally:

```bash
cd docs
make html # Generate HTML documentation
```

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt)
file for details.
