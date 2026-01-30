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
#include <memory>
#include <unordered_map>
#include <string>
#include <optional>

 // Represents a 2D keypoint with position and confidence score
struct Point2D
{
    float x = 0.0;     
    float y = 0.0;      
    float score = 0.0;  
};

/**
 * Defines where pose keypoints come from.
 * - OpenPoseLive: run OpenPose per frame (slow)
 * - Precomputed: load joints from JSON (fast)
 */
 enum class PoseSource {
    OpenPoseLive,
    Precomputed
 };

// Wraps the OpenPose API
// Input: OpenCV frame
// Output: 2D detected joints for a single person.
class PoseDetector {
public:
    // Constructor with optional path for precomputed keypoints
    explicit PoseDetector(std::optional<std::string> precomputedKeypointsPath);

    // Load keypoints for Precomputed mode
    bool loadKeypoints(const std::string& jsonPath);

    /**
     * Detect or retrieve keypoints for a frame.
     * @param frame     Current video frame (ignored in Precomputed mode)
     * @param frameIdx  Frame index (used to lookup keypoints)
     */
    std::vector<Point2D> detect(const cv::Mat& frame, int frameIdx);

private:
    PoseSource source_;

    // Precomputed mode
    // frame index -> list of 2D joints (x,y)
    std::unordered_map<int, std::vector<Point2D>> cachedPoses_;

    // OpenPose live mode
    std::unique_ptr<op::Wrapper> openpose_;
};
