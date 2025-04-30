# MIT License
#
# Copyright (C) 2025 Goodarz Mehr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

# BEVFusion Docker Configuration Script
#
# This script performs all the necessary steps for creating a BEVFusion Docker
# image.
#
# The base Docker image is Ubuntu 20.04 with CUDA 11.8. If you want to use a
# different base image, you may need to modify "ubuntu2004/x86_64" when
# fetching keys, based on your Ubuntu release and system architecture.

# Build Arguments (Case Sensitive):
#
# USER: username inside each container, set to "bf" by default.

# Installation:
#
# 1. Install Docker on your system (https://docs.docker.com/engine/install/).
# 2. Install the Nvidia Container Toolkit
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).
# It exposes your Nvidia graphics card to Docker containers.
# 3. Install the Nvidia Container Runtime and set it as the default runtime
# (https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).
# 4. In the Dockerfile directory, run
#
# docker build --no-cache --rm --build-arg ARG -t bevfusion:develop .

# Usage:
#
# Launch a container by running
#
# docker run --privileged --gpus all --network=host -e DISPLAY=$DISPLAY
# -v [path/to/BEVFusion]:/home/bevfusion
# -v [path/to/dataset]:/dataset
# --shm-size 32g -it bevfusion:develop /bin/bash
#
# Use "nvidia-smi" to ensure your graphics card is visible inside the
# container. Then, as the root user inside the "bevfusion" directory, run
#
# python setup.py develop

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Define build arguments and environment variables.

ARG USER=bf

ENV USER=${USER}
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere;Ada;Hopper"
ENV WITH_CUDA=1
ENV USE_CUDA=1
ENV USE_CUDNN=1

# Add new user and install prerequisite packages.

WORKDIR /home

RUN useradd -m ${USER}

RUN set -xue && apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
&& apt update \
&& apt install -y build-essential cmake debhelper git wget libsdl2-2.0-0 libsm6 libgl1-mesa-glx libomp5 pip unzip \
libjpeg8 libtiff5 software-properties-common nano fontconfig g++ gcc libglib2.0-0 openmpi-bin openmpi-common \
libopenmpi-dev libjpeg-dev libgtk2.0-dev libnvidia-gl-550 libnvidia-common-550 python-is-python3

RUN pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 pillow==8.4.0 \
mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0 mpi4py==3.0.3 numba==0.48.0 numpy==1.23.5 cmake==3.22.6 future==0.18.2 \
yapf==0.40.1 mkl==2022.1.0 mkl-include==2022.1.0 astunparse nuscenes-devkit tqdm torchpack matplotlib opencv-python \
open3d ninja dataclasses -f https://download.pytorch.org/whl/torch_stable.html

# Uninstall MMCV and PyTorch.

RUN pip uninstall mmcv mmcv-full -y
RUN pip uninstall torch torchvision torchaudio -y

# Build and install PyTorch from source.

WORKDIR /opt

RUN git clone --recursive --branch pytorch-from-1.11.0-cuda-12.1 https://github.com/GoodarzMehr/pytorch.git

WORKDIR /opt/pytorch

ENV CMAKE_PREFIX_PATH="$(dirname $(which python3))/../share/cmake"

RUN FORCE_CUDA=1 python setup.py install
RUN FORCE_CUDA=1 python setup.py develop

WORKDIR /opt

# Build and install TorchVision from source.

RUN git clone --recursive --branch v0.12.0 https://github.com/pytorch/vision.git

WORKDIR /opt/vision

RUN FORCE_CUDA=1 python setup.py install
RUN FORCE_CUDA=1 python setup.py develop

WORKDIR /home

# Install MMCV.

RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install \
--no-cache-dir "git+https://github.com/GoodarzMehr/mmcv.git@bevfusion-from-1.4.0"

# Install PyTorch3D.

RUN FORCE_CUDA=1 pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"

USER ${USER}
