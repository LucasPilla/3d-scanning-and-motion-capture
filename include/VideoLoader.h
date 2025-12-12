// VideoLoader
// ------------
// - Responsible ONLY for reading frames from a video.
// - Hides OpenCV's VideoCapture from the rest of the system.


#pragma once
#include <opencv2/opencv.hpp>
#include <string>


class VideoLoader {
public:
    VideoLoader(const std::string& path);

    bool isOpen() const;
    bool readFrame(cv::Mat& frame);

    double fps() const;
    int width() const;
    int height() const;

private:
    cv::VideoCapture cap;
};
