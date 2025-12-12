// Visualization.cpp
// Implements visualization utilities for:
//   - rendering 3D SMPL meshes
//   - drawing keypoints on video frames
//   - exporting final videos / images
//
// TODOs:
//   - display 3D mesh using OpenGL or save as OBJ
//   - overlay SMPL skeleton on the RGB video
//   - export sequences as mp4/avi for the final demo

#include "Visualization.h"

Visualization::Visualization(int width, int height, double fps)
{
    writer.open(
        "output.avi",
        cv::VideoWriter::fourcc('M','J','P','G'),
        fps,
        cv::Size(width, height)
    );
}

void Visualization::write(const cv::Mat& frame)
{
    if (writer.isOpened()) {
        writer.write(frame);
    }
}
