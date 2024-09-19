ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=22.04
ARG USER_ID=u29u30

# Define base image.
# 1. Using from NVIDIA  
# [ref] https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2023
# [ref] https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# FROM nvcr.io/nvidia/pytorch:23.10-py3
# 2. Building from scratch 
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}
# 3. Exisitng image
# FROM nvsf_nerf:v1

# metainformation
# LABEL org.opencontainers.image.version = "0.1.0"

# Variables used at build time.
# CUDA architectures, required by Colmap and tiny-cuda-nn.
# NOTE: Most commonly used GPU architectures are included and supported here. To speedup the image build process remove all architectures but the one of your explicit GPU. 
# Find details here: https://developer.nvidia.com/cuda-gpus (8.6 translates to 86 in the line below) or in the docs.
# ARG CUDA_ARCHITECTURES=90;89;86;80;75;61
ARG CUDA_ARCHITECTURES=61

# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libcgal-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libglib2.0-0 \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    ninja-build \
    protobuf-compiler \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    qtbase5-dev \
    vim-tiny \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install --reinstall -y \
    libmpich-dev \
    hwloc-nox \
    libmpich12 \
    mpich

# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Certificates for HPC
ADD CA.crt /usr/local/share/ca-certificates/CA.crt
RUN echo CA.crt >> /etc/ca-certificates.conf
RUN chmod 644 /usr/local/share/ca-certificates/CA.crt && update-ca-certificates

# Upgrade pip and install packages.
RUN python3.10 -m pip install --no-cache-dir --upgrade pip "setuptools<70.0" pathtools promise pybind11
SHELL ["/bin/bash", "-c"]

# Install pytorch and submodules
RUN CUDA_VER=${CUDA_VERSION%.*} && CUDA_VER=${CUDA_VER//./} && python3.10 -m pip install --no-cache-dir \
    torch==2.1.2+cu${CUDA_VER} \
    torchvision==0.16.2+cu${CUDA_VER} \
    --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

# Install tynyCUDNN (we need to set the target architectures as environment variable first).
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

# Change working directory
WORKDIR /my_workspace

# Copy files to working dir folder
COPY . .

# [Debug] Check if files are there
# RUN ls -alh /my_workspace

# Install requirements
RUN python3.10 -m pip install -r requirements.txt

# Install torch extensions
RUN python3.10 -m pip install \
    nvsf/nerf/raymarching \
    nvsf/nerf/chamfer3D

# Install nvsf
RUN pip install -e .

# # Make sure nvsf is installed
RUN python -c "import nvsf; print(nvsf.__version__)"

# Bash as default entrypoint.
CMD /bin/bash -l