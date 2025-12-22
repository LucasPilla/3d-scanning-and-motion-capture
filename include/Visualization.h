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
#include <string>
#include "PoseDetector.h"
#include "SMPLModel.h" // Needed for SMPLMesh struct

class Visualization {
public:
    Visualization(int width, int height, double fps);

    void write(const cv::Mat& frame);
    void drawKeypoints(cv::Mat& frame, const std::vector<Point2D>& keypoints);
    
    // New functionality for Issue #12
    void exportMesh(const SMPLMesh& mesh, int frameIdx);

private:
    cv::VideoWriter writer;
    std::string outputDir = "debug_meshes"; // Default directory for OBJ exports
};