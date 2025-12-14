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
        "output.mp4",
        cv::VideoWriter::fourcc('a','v','c','1'),
        fps,
        cv::Size(width, height),
        true
    );

    if (!writer.isOpened()) {
        std::cerr << "ERROR: Could not open video writer!" << std::endl;
    } else {
        std::cout << "Video writer opened. " << std::endl;
    }
}

void Visualization::write(const cv::Mat& frame)
{
    if (!frame.empty() &&writer.isOpened()) {
        writer.write(frame);
    }
}
