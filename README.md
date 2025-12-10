# 3d-scanning-and-motion-capture

[![Open Proposal](https://img.shields.io/badge/Proposal-PDF-red?style=flat-square)](https://github.com/LucasPilla/3d-scanning-and-motion-capture/docs/proposal.pdf)

## Environment

First, navigate to the folder containing your `Dockerfile` and build the image. This will create a Docker image named `3dsmc-project`.

```
docker build -t 3dsmc-project .
```

Then, run the container using the command below. This mounts your current directory to `/usr/src/project` inside the container so you can access your code and video files.

```
docker run -it --rm -v $(pwd):/usr/src/project 3dsmc-project
```

## Build 

Once inside the container, build the project using CMake:

```
mkdir build
cd build
cmake ..
make
```

## Run

The current pipeline processes a video frame by frame, runs 2D keypoint detection with OpenPose, and saves the results as a new video file.

```
./pipeline <path-to-video>
```




