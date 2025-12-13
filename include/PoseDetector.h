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

// Wraps the OpenPose API
// Input: OpenCV frame
// Output: flat vector of keypoint floats [x0,y0,score0, x1,y1,score1, ...]
//         for a single selected person (highest-confidence), in OpenPose's
//         body-part order. Joints with very low confidence are zeroed out.

class PoseDetector {
public:
    PoseDetector();

    std::vector<float> detect(const cv::Mat& frame);

private:
    std::unique_ptr<op::Wrapper> wrapper;
};
