# DIS Project

This repository is for DIS project 12: Gaussian Processes for Latent force Models.


## How to install & run the project

The code for this project can either be installed and run within a docker container, or the repository can be run locally. To clone the repository run the following command:

```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/projects/wp289.git
```

### Installing and running in Docker

TODO: add more info about docker and pasting images

A dockerfile is provided to run the project alongside all of it's dependencies (defined in environment.yml) in a docker container. To build the docker image from the provided dockerfile run the following commands after cloning the repository:

```bash
docker build -t lfm_image . # builds docker image
```

As the project requires command-line interactions, the docker container needs to be run in interactive mode:

```bash
docker run -it lfm_image
```

The project can then be ran within the docker container.

### Running locally

To run this project without docker, create a `conda` environment from the provided `environment.yml` file and activate it:

```bash
conda env create -f environment.yml # create env
conda activate gpjax_wp289          # activate env
```

## How to use the project

TODO


### Documentation

TODO: discuss building documentation (seperate gpjax and gpytorch)

## License

This project is licensed under the MIT license - see the [LICENSE](license.txt)
file for details.

### A note on the use of generation tools

TODO
