// Visualization
// --------------
// Responsible for rendering and exporting results.
// Responsibilities:
//  - Visualize SMPL mesh in 3D
//  - Overlay SMPL joints or skeleton on original frames
//  - Save output videos or image sequences
// Used after:
//  - FittingOptimizer (baseline visualization)
//  - TemporalSmoother (smoothed final output)


#pragma once
#include <opencv2/opencv.hpp>


class Visualization {
public:
    Visualization(int width, int height, double fps);

    void write(const cv::Mat& frame);

private:
    cv::VideoWriter writer;
};
