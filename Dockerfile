FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

##############################
######    Dependencies   #####
##############################

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    protobuf-compiler \
    libboost-all-dev \
    libhdf5-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libeigen3-dev \
    libomp-dev \
    python3 \
    python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

##############################
##### Build Ceres 2.2.0 ######
##############################

WORKDIR /opt

RUN git clone https://ceres-solver.googlesource.com/ceres-solver \
 && cd ceres-solver \
 && git checkout 2.2.0 \
 && mkdir build

WORKDIR /opt/ceres-solver/build

ENV CXXFLAGS="-O3 -march=native"
ENV CFLAGS="-O3 -march=native"

RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCERES_THREADING_MODEL=OPENMP \
    -DCERES_USE_OPENMP=ON \
    -DLAPACK=ON \
    -DBLAS=ON
 
RUN make -j2 && make install

##############################
###### Build OpenPose ########
##############################

WORKDIR /opt
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

WORKDIR /opt/openpose
RUN git submodule update --init --recursive

WORKDIR /opt/openpose/build

RUN cmake .. \
    -DGPU_MODE=CPU_ONLY \
    -DBUILD_PYTHON=OFF \
    -DDOWNLOAD_BODY_25_MODEL=ON \
    -DDOWNLOAD_FACE_MODEL=OFF \
    -DDOWNLOAD_HAND_MODEL=OFF \
    -DCMAKE_BUILD_TYPE=Release

RUN make -j2 && make install

##############################
###### Runtime Settings ######
##############################

ENV LD_LIBRARY_PATH="/usr/local/lib"
ENV OMP_NUM_THREADS=8
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=cores

##############################
###### Python Deps ##########
##############################

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /usr/src/project
CMD ["bash"]
