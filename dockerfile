# Use basic miniconda image
FROM continuumio/miniconda3

# Create project directory
RUN mkdir LFM

# Copy project files
COPY . ./LFM
WORKDIR /LFM

# Create conda environment
RUN conda env create --name gpjax_wp289 --file environment.yml

# Activate conda environment
RUN echo "conda activate gpjax_wp289" >> ~/.bashrc

# Change shell to bash with login
SHELL ["/bin/bash", "--login", "-c"]
