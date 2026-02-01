// VideoLoader.cpp
// Implementation of the VideoLoader class declared in VideoLoader.h.

#include "VideoLoader.h"
#include <iostream>

VideoLoader::VideoLoader(const std::string& path)
{
    cap.open(path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video: " << path << "\n";
        exit(0);
    }
}

bool VideoLoader::isOpen() const { return cap.isOpened(); }
bool VideoLoader::readFrame(cv::Mat& frame) { return cap.read(frame); }

double VideoLoader::fps() const { return cap.get(cv::CAP_PROP_FPS); }
int VideoLoader::width() const { return (int)cap.get(cv::CAP_PROP_FRAME_WIDTH); }
int VideoLoader::height() const { return (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT); }
