# 3d-scanning-and-motion-capture

[![Open Proposal](https://img.shields.io/badge/Proposal-PDF-red?style=flat-square)](https://github.com/LucasPilla/3d-scanning-and-motion-capture/blob/main/docs/proposal.pdf)
[![Open Final Report](https://img.shields.io/badge/Final%20Report-PDF-red?style=flat-square)](https://github.com/LucasPilla/3d-scanning-and-motion-capture/blob/main/docs/report.pdf)

## TODO

- [x] Configure repository, CMake and Docker
- [x] Implement 2D keypoint detection with OpenPose
- [x] Implement SMPL parameters optimization with Ceres
- [x] Implement proposals
- [x] Evaluate baseline
- [x] Evaluate proposals
- [x] Write the final report
- [x] Prepare presentation

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

## Data

We do not provide that data in this repository, according to license restrictions.

To run this application, you need to download the **SMPL model**, **GMM**, and **OpenPose joint regressor** files:

- **SMPL model**:  
  Download from [https://smpl.is.tue.mpg.de/download.php](https://smpl.is.tue.mpg.de/download.php).  
  This link provides SMPL model files (`.pkl`) for both **male** and **female** bodies.

- **GMM**:  
  Download from [https://smplify.is.tue.mpg.de/download.php](https://smplify.is.tue.mpg.de/download.php).  
  This link provides the GMM file (`.pkl`).

- **OpenPose joint regressor**:  
  Download from EasyMocap repository [https://github.com/zju3dv/EasyMocap/tree/master/data/smplx](https://github.com/zju3dv/EasyMocap/tree/master/data/smplx).
  The file is named `J_regressor_body25.npy`.

These files are required to run the `preprocess.py` script located in the `scripts` folder.  
The script generates a JSON file containing all model data, which is then passed as a parameter to the pipeline.

To run the preprocess step you must install the dependencies from `requirements.txt` and execute: 

```text
usage: preprocess.py [-h] --model_path MODEL_PATH --save_dir SAVE_DIR --output_name OUTPUT_NAME --gmm GMM --openpose_joint_regressor
                     OPENPOSE_JOINT_REGRESSOR [--capsule_regressor_path CAPSULE_REGRESSOR_PATH]

Preprocess SMPL models into JSON.

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the raw SMPL model (pickle)
  --save_dir SAVE_DIR   Directory to save the processed output
  --output_name OUTPUT_NAME
                        Name of the output JSON file (e.g., smpl_model.json)
  --gmm GMM             Path to the GMM prior (pickle)
  --openpose_joint_regressor OPENPOSE_JOINT_REGRESSOR
                        Path to the OpenPose joint regressor (.npy)
  --capsule_regressor_path CAPSULE_REGRESSOR_PATH
                        Path to the capsule regressors .npz file (Optional)
```

For example, considering that all data is download within `models` folder:

```bash
  python3 scripts/preprocess.py  \
    --model_path ./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl \
    --gmm ./models/gmm_08.pkl \
    --openpose_joint_regressor ./models/J_regressor_body25.npy \
    --save_dir ./models/ \
    --output_name smpl_male.json
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
