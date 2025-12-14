// PoseDetector
// -------------
// Wraps OpenPose for extracting 2D human pose keypoints from video frames.
// Responsibilities:
//  - Initialize OpenPose with required models
//  - Process each frame with OpenPose
//  - Output 2D keypoints as a simple array/vector
// Used by:
//  - FittingOptimizer (which fits SMPL to the 2D joints)

#pragma once
#include <openpose/headers.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// Represents a 2D keypoint with position and confidence score
struct Point2D
{
    float x = 0.0;     
    float y = 0.0;      
    float score = 0.0;  
};

// Represents the 2D pose for a single person
struct Pose2D
{
    std::vector<Point2D> keypoints;
};

// Wraps the OpenPose API
// Input: OpenCV frame
// Output: 2D pose for a single person (highest-confidence). 
//         Joints with very low confidence are zeroed out.
class PoseDetector {
public:
    PoseDetector();

    Pose2D detect(const cv::Mat& frame);

private:
    std::unique_ptr<op::Wrapper> wrapper;
};
