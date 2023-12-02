# MIT License
#
# Copyright (C) 2023 Goodarz Mehr
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
# Performs all the necessary tasks for the generation of a BEVFusion Docker
# image containing all the required dependencies.
#
# The base Docker image is Ubuntu 20.04 with CUDA 11.3. If you want to use a
# different base image, you may need to modify "ubuntu2004/x86_64" when
# fetching keys, according to your Ubuntu release and system architecture.

# Build Arguments (Case Sensitive):
#
# USER:            default username inside each container, set to "bf"
#                  by default.

# Installation:
#
# 1. Install Docker on your system (https://docs.docker.com/engine/install/).
# 2. Install the Nvidia Container Toolkit
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).
# It exposes your Nvidia graphics card to Docker containers.
# 3. In the Dockerfile directory, run
#
# docker build --no-cache --rm --build-arg ARG -t bevfusion:develop .

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Define build arguments and environment variables.

ARG USER=bf

ENV USER=${USER}
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive

# Add new user and install prerequisite packages.

WORKDIR /home

RUN useradd -m ${USER}

RUN set -xue && apt-key del 7fa2af80 \
&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
&& apt update \
&& apt install -y build-essential cmake debhelper git wget libsdl2-2.0-0 libsm6 libgl1-mesa-glx libomp5 pip unzip \
libjpeg8 libtiff5 software-properties-common nano fontconfig g++ gcc libglib2.0-0 openmpi-bin openmpi-common \
libopenmpi-dev libgtk2.0-dev libnvidia-gl-520=520.61.05-0ubuntu1 libnvidia-common-520=520.61.05-0ubuntu1 \
python-is-python3

RUN pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 pillow==8.4.0 tqdm \
torchpack mmcv==1.4.5 mmcv-full==1.4.5 mmdet==2.20.0 nuscenes-devkit mpi4py==3.0.3 numba==0.48.0 numpy==1.23.5 \
yapf==0.40.1 matplotlib opencv-python open3d ninja -f https://download.pytorch.org/whl/torch_stable.html

RUN pip uninstall mmcv mmcv-full -y
RUN pip install --no-cache-dir mmcv==1.4.0 mmcv-full==1.4.0

USER ${USER}
