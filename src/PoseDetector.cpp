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

    // Use smaller network input for speed (width, height in pixels)
    // You can tweak these: smaller = faster but less accurate
    poseConfig.netInputSize = op::Point<int>{320, 176};  // e.g. 320x176

    // Disable rendering overlays to save CPU (we don't use the rendered image)
    poseConfig.renderMode = op::RenderMode::None;

    wrapper = std::make_unique<op::Wrapper>(op::ThreadManagerMode::Asynchronous);
    wrapper->configure(poseConfig);
    wrapper->start(); // Start the OpenPose processing threads. Must be called before detecting any frames
}

// Extract 2D joints from a video frame using OpenPose.
std::vector<float> PoseDetector::detect(const cv::Mat& frame)
{
    auto toProcess = OP_CV2OPCONSTMAT(frame);          // Convert OpenCV image into OpenPose input format
    auto result    = wrapper->emplaceAndPop(toProcess); // Send frame to OpenPose and retrieve output

    if (!result || result->empty()) {
        return {}; // No detections
    }

    const auto& kp = result->at(0)->poseKeypoints;

    // OpenPose stores results in poseKeypoints:
    //   shape = [numPeople, numBodyParts, 3]
    //   data  = (x, y, score)
    const auto numPeople     = kp.getSize(0);
    const auto numBodyParts  = kp.getSize(1);
    const auto numComponents = kp.getSize(2); // should be 3

    if (numPeople == 0 || numBodyParts == 0 || numComponents < 3) {
        return {};
    }

    constexpr float kMinJointScore = 0.1f; // TODO: tune this threshold

    // 1) Pick highest-confidence person
    int   bestPerson   = -1;
    float bestScoreSum = 0.0f;

    for (int person = 0; person < numPeople; ++person) {
        float scoreSum = 0.0f;
        for (int part = 0; part < numBodyParts; ++part) {
            int baseIndex = 3 * (person * numBodyParts + part);
            float score   = kp[baseIndex + 2];
            if (score >= kMinJointScore) {
                scoreSum += score;
            }
        }
        if (scoreSum > bestScoreSum) {
            bestScoreSum = scoreSum;
            bestPerson   = person;
        }
    }

    if (bestPerson == -1) {
        return {}; // no person with joints above threshold
    }

    // 2) Extract that person's joints, zeroing low-confidence joints
    std::vector<float> keypoints(numBodyParts * 3, 0.0f);

    for (int part = 0; part < numBodyParts; ++part) {
        int baseIndex = 3 * (bestPerson * numBodyParts + part);
        float x       = kp[baseIndex + 0];
        float y       = kp[baseIndex + 1];
        float score   = kp[baseIndex + 2];

        if (score < kMinJointScore) {
            x = 0.0f;
            y = 0.0f;
            score = 0.0f;
        }

        int outIndex = 3 * part;
        keypoints[outIndex + 0] = x;
        keypoints[outIndex + 1] = y;
        keypoints[outIndex + 2] = score;
    }

    // Joints remain in OpenPose BODY_25 order; we’ll align this with SMPL later
    return keypoints;
}