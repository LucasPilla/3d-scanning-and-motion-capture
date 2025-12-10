# Use an official Ubuntu base image
FROM ubuntu:22.04

# Set environment variable to avoid user prompts while building docker image
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    protobuf-compiler \
    libboost-all-dev \
    libhdf5-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev

##############################
####### Install Eigen ########
##############################

RUN apt install -y libeigen3-dev

##############################
###### Install Ceres ######
##############################

RUN apt-get install -y libceres-dev

##############################
####### Install OpenCV #######
##############################

RUN apt-get install -y libopencv-dev

##############################
###### Install OpenPose ######
##############################

# Clone repository from github
WORKDIR /opt
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

# Fetch submodules
WORKDIR /opt/openpose
RUN git submodule update --init --recursive --remote

# Create build directory
WORKDIR /opt/openpose/build

# Configure CMake
# NOTE: We use GPU_MODE=CPU_ONLY to ensure the Docker build succeeds on machines 
# without NVIDIA drivers mapped. If you have CUDA, change this to CUDA.
RUN cmake .. \
    -DGPU_MODE=CPU_ONLY \
    -DBUILD_PYTHON=OFF \
    -DDOWNLOAD_BODY_25_MODEL=ON \
    -DDOWNLOAD_FACE_MODEL=OFF \
    -DDOWNLOAD_HAND_MODEL=OFF

# Build
RUN make -j$(nproc) && make install

# Add /usr/local/lib to PATH as some libraries are installed there.
ENV LD_LIBRARY_PATH="/usr/local/lib"

# Initial directory
WORKDIR /usr/src/project

# Default command
CMD ["bash"]

