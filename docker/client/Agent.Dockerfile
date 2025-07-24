# FROM python:3.9.21-slim

# WORKDIR /home/wato_research

# COPY . /home/wato_research

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# # Install dependencies
# RUN python3.9 -m pip install -r requirements.txt

# CMD /bin/bash

##########################################
# Dockerfile for TD-MPC2                 #
# Nicklas Hansen, 2023 (c)               #
# https://www.tdmpc2.com                 #
# -------------------------------------- #
# Build instructions:                    #
# docker build . -t <user>/tdmpc2:1.0.1  #
# docker push <user>/tdmpc2:1.0.1        #
# -------------------------------------- #
# Run:                                   #
# docker run -i \                        #
#   -v <path>/<to>/tdmpc2:/tdmpc2 \      #
#   --gpus all \                         #
#   -t <user>/tdmpc2:1.0.1 \             #
#   /bin/bash                            #
##########################################

# base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git nano rsync vim tree curl wget \
    swig ffmpeg unzip htop tmux xvfb ca-certificates bash-completion libjpeg-dev libpng-dev \
    libssl-dev libcurl4-openssl-dev libopenmpi-dev zlib1g-dev qtbase5-dev qtdeclarative5-dev \
    libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 libgl1-mesa-glx libosmesa6 \
    libosmesa6-dev libglew-dev mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# conda environment
# COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
# COPY environment.yaml /root
# RUN conda update conda && \
#     conda env update -n base -f /root/environment.yaml && \
#     rm /root/environment.yaml && \
#     conda clean -ya && \
#     pip cache purge && \
#     conda init
# --------------------------- TD-MPC2 ----------------------------
COPY docker/client/requirements.txt /root
RUN pip install -r /root/requirements.txt && \
    pip cache purge && \
    rm /root/requirements.txt

# Fix moviepy issue in wandb
RUN sed -i 's/moviepy\.editor/moviepy/g' /opt/conda/lib/python3.11/site-packages/wandb/sdk/data_types/video.py

WORKDIR /root/tdmpc2

COPY ./src/client/tdmpc2/tdmpc2 .
COPY ./src/client/rlenv.py .
COPY ./src/client/client.py .

SHELL ["/bin/bash", "-c"]
RUN echo "cd /root/tdmpc2" >> /root/.bashrc
# image does not include metaworld, maniskill, myosuite
# these can be installed separately; see environment.yaml for details

# success!
RUN echo "Successfully built TD-MPC2 Docker image!"

