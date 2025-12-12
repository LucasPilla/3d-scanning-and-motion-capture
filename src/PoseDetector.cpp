// PoseDetector.cpp
// Implementation of the PoseDetector class (OpenPose wrapper).
// Contains the code for:
//   - initializing the OpenPose wrapper
//   - running pose estimation on each frame
//   - converting OpenPose output to our internal keypoint format
//
// Main TODOs:
//   - configure OpenPose model paths
//   - implement detectKeypoints(frame)
//   - handle multiple people (for now we will use only person 0)

#include "PoseDetector.h"
#include <iostream>

PoseDetector::PoseDetector()
{
    // Configure OpenPose model parameters
    op::WrapperStructPose poseConfig;
    poseConfig.modelFolder = "/opt/openpose/models/";
    poseConfig.renderMode = op::RenderMode::Cpu;

    wrapper = std::make_unique<op::Wrapper>(op::ThreadManagerMode::Asynchronous);
    wrapper->configure(poseConfig);
    wrapper->start(); // Start the OpenPose processing threads. Must be called before detecting any frames
}

    // Extract 2D joints from a video frame using OpenPose.
std::vector<float> PoseDetector::detect(const cv::Mat& frame)
{
    auto toProcess = OP_CV2OPCONSTMAT(frame); // Convert OpenCV image into OpenPose input format
    auto result = wrapper->emplaceAndPop(toProcess); // Send frame to OpenPose and retrieve output

    if (!result || result->empty()) {
        return {};
    }

    const auto& kp = result->at(0)->poseKeypoints;

    // OpenPose stores results in poseKeypoints:
    // shape = [numPeople, numBodyParts, 3]
    // data = (x, y, score)

    std::vector<float> keypoints(kp.getVolume());
    for (int i = 0; i < kp.getVolume(); i++) keypoints[i] = kp[i];

    return keypoints;
}
