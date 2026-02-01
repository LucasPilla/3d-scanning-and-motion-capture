# 3d-scanning-and-motion-capture

[![Open Proposal](https://img.shields.io/badge/Proposal-PDF-red?style=flat-square)](https://github.com/LucasPilla/3d-scanning-and-motion-capture/blob/main/docs/proposal.pdf)

## TODO

- [x] Configure repository, CMake and Docker
- [x] Implement 2D keypoint detection with OpenPose
- [x] Implement SMPL parameters optimization with Ceres
- [x] Implement proposals
- [ ] Evaluate baseline
- [ ] Evaluate proposals
- [ ] Write the final report
- [ ] Prepare presentation

## Setup

First, clone this GitHub repository.

```
git clone https://github.com/LucasPilla/3d-scanning-and-motion-capture
```

Navigate to the folder containing the `Dockerfile` and build the image. This will create a Docker image named `3dsmc-project`.

```
docker build -t 3dsmc-project .
```

Then, run the container using the command below. This mounts your current directory to `/usr/src/project` inside the container so you can access your code and video files.

```
docker run -it --rm -v $(pwd):/usr/src/project 3dsmc-project
```

## SMPL

To run this application, you need to download the **SMPL model** and the **GMM** files:

- **SMPL model**:  
  Download from [https://smpl.is.tue.mpg.de/download.php](https://smpl.is.tue.mpg.de/download.php).  
  This link provides SMPL model files (`.pkl`) for both **male** and **female** bodies.

- **GMM**:  
  Download from [https://smplify.is.tue.mpg.de/download.php](https://smplify.is.tue.mpg.de/download.php).  
  This link provides the GMM file (`.pkl`).

These files are required to run the `preprocess.py` script located in the `scripts` folder.  
The script generates a JSON file containing all model data, which is then passed as a parameter to the pipeline.

## Build 

Once inside the container, build the project using CMake:

```
mkdir build
cd build
cmake ..
make
```

## Run

```text
Usage: pipeline [--help] [--version] --video-path VAR --smpl-path VAR [--output VAR] [--precomputed-keypoints VAR] [--frame VAR] [--debug] [--skip-viz]

Optional arguments:
  -h, --help               shows help message and exits 
  -v, --version            prints version information and exits 
  --video-path             Path to video file [required]
  --smpl-path              Path to SMPL model (.json) [required]
  --output                 Output folder [nargs=0..1] [default: "./output"]
  --precomputed-keypoints  Path to pre-computed keypoints (.json) 
  --frame                  Process only this frame 
  --debug                  Save debug data as a JSON 
  --skip-viz               Skip visualization
```

Example:

```bash
./pipeline \
  --video-path ../data/sample.avi \
  --smpl-path ../models/smpl_female.json \
  --precomputed-keypoints ../data/keypoints.json \
  --debug
```
