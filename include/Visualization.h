// Visualization
// --------------
// Responsible for rendering and exporting results.

// Responsibilities:
//  - Overlay OpenPose joints on original frames
//  - Overlay SMPL joints on original frames
//  - Visualize SMPL mesh in 3D
//  - Save output videos or image sequences

// Used after:
//  - FittingOptimizer (baseline visualization)
//  - TemporalSmoother (smoothed final output)


#pragma once
#include <opencv2/opencv.hpp>
#include "PoseDetector.h"


class Visualization {
public:
    Visualization(int width, int height, double fps);

    void write(const cv::Mat& frame);
    void drawKeypoints(cv::Mat& frame, const std::vector<Point2D>& keypoints);

    // generic 2D joint drawing (for SMPL projections, etc.)
    void drawJoints(cv::Mat& frame,
        const std::vector<Point2D>& joints,
        const cv::Scalar& color,
        int radius = 4);

private:
    cv::VideoWriter writer;
};
