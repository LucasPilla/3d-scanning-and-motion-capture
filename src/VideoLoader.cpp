// VideoLoader.cpp
// Implementation of the VideoLoader class declared in VideoLoader.h.
// This file contains the actual logic for:
//   - opening the video file using OpenCV
//   - reading frames sequentially
//   - optional frame skipping / resizing
//   - returning frames for processing in the pipeline
//
// TODOs inside this file should implement:
//   - constructor logic
//   - nextFrame() and hasNextFrame()
//   - any preprocessing (resizing, normalization)

#include "VideoLoader.h"
#include <iostream>

VideoLoader::VideoLoader(const std::string& path)
{
    cap.open(path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video: " << path << "\n";
    }
}

bool VideoLoader::isOpen() const { return cap.isOpened(); }
bool VideoLoader::readFrame(cv::Mat& frame) { return cap.read(frame); }

double VideoLoader::fps() const { return cap.get(cv::CAP_PROP_FPS); }
int VideoLoader::width() const { return (int)cap.get(cv::CAP_PROP_FRAME_WIDTH); }
int VideoLoader::height() const { return (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT); }
