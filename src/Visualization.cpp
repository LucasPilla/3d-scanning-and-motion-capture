// Visualization.cpp

// Implements visualization utilities for:
//  - Overlay OpenPose joints on original frames
//  - Overlay SMPL joints on original frames
//  - Visualize SMPL mesh in 3D
//  - Save output videos or image sequences

// TODOs:
//  - Overlay OpenPose joints on original frames (done)
//  - Overlay SMPL joints on original frames
//  - Visualize SMPL mesh in 3D
//  - Save output videos or image sequences (done)

#include "Visualization.h"
#include "PoseDetector.h"

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

void Visualization::drawKeypoints(cv::Mat& frame, const std::vector<Point2D>& keypoints)
{
    // This function expects 25 keypoints, according to BODY_25 model
    if (keypoints.size() != 25) {
        return;
    }

    // BODY_25 keypoint connections (skeleton)
    // Each pair represents a connection between two keypoints
    const std::vector<std::pair<int, int>> connections = {
        {1, 8},                         // Neck → MidHip
        {1, 2}, {2, 3}, {3, 4},         // Right arm
        {1, 5}, {5, 6}, {6, 7},         // Left arm
        {8, 9}, {9, 10}, {10, 11},      // Right leg
        {8, 12}, {12, 13}, {13, 14},    // Left leg
        {1, 0},                         // Neck → Nose
        {0, 15}, {15, 17},              // Right face
        {0, 16}, {16, 18},              // Left face
        {14, 19}, {19, 20}, {14, 21},   // Left foot
        {11, 22}, {22, 23}, {11, 24}    // Right foot
    };

    
    // Colors for visualization
    const cv::Scalar skeleton_color = cv::Scalar(0, 255, 0);    // Green for skeleton
    const cv::Scalar keypoint_color = cv::Scalar(0, 0, 255);    // Red for keypoints
    
    const int keypoint_radius = 4;
    const int skeleton_thickness = 2;
    
    // Draw skeleton connections
    for (const auto& connection : connections) {

        int idx1 = connection.first;
        int idx2 = connection.second;
        
        const Point2D pt1 = keypoints[idx1];
        const Point2D pt2 = keypoints[idx2];
        
        if (
            pt1.x > 0 && pt1.y > 0 && pt1.score > 0.0 &&
            pt2.x > 0 && pt2.y > 0 && pt2.score > 0.0
        ) {
            cv::line(frame, cv::Point(pt1.x, pt1.y), cv::Point(pt2.x, pt2.y), skeleton_color, skeleton_thickness);
        }
    }
    
    // Draw keypoints
    for (size_t i = 0; i < keypoints.size(); ++i) {

        const Point2D pt = keypoints[i];

        // Skip invalid keypoints
        if (pt.x <= 0 || pt.y <= 0 || pt.score <= 0.0) {
            continue;
        }
        
        cv::circle(frame, cv::Point(pt.x, pt.y), keypoint_radius, keypoint_color, -1);
    }
}