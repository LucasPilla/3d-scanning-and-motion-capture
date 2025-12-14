// PoseDetector.cpp
// Implementation of the PoseDetector class (OpenPose wrapper).
// 
// Contains the code for:
//   - initializing the OpenPose wrapper
//   - running pose estimation on each frame
//   - converting OpenPose output to our internal keypoint format
//
// TODOs:
//   - Set OpenPose configs (done)
//   - Implement detect keypoints perframe (done)
//   - Handle multiple people (for now we will use person at index 0)

#include "PoseDetector.h"
#include <iostream>

PoseDetector::PoseDetector()
{
    // Configure OpenPose model parameters
    op::WrapperStructPose poseConfig;
    poseConfig.modelFolder = "/opt/openpose/models/";

    // Use smaller network input for speed (width, height) in pixels
    // You can tweak these: smaller = faster but less accurate
    poseConfig.netInputSize = op::Point<int>{320, 176};

    // Disable rendering overlays to save CPU (we don't use the rendered image)
    poseConfig.renderMode = op::RenderMode::None;

    wrapper = std::make_unique<op::Wrapper>(op::ThreadManagerMode::Asynchronous);
    wrapper->configure(poseConfig);

    // Start the OpenPose processing threads. Must be called before detecting any frames
    wrapper->start(); 
}

// Extract 2D joints from a video frame using OpenPose.
Pose2D PoseDetector::detect(const cv::Mat& frame)
{
    Pose2D pose2D;

    // Convert OpenCV image into OpenPose input format
    auto toProcess = OP_CV2OPCONSTMAT(frame);    

    // Send frame to OpenPose and retrieve output   
    auto result = wrapper->emplaceAndPop(toProcess); 

    if (!result || result->empty()) {
        return pose2D; // No detections
    }

    const auto& kp = result->at(0)->poseKeypoints;

    // OpenPose stores results in poseKeypoints.
    // It is a flatted array [x, y, conf, x, y, conf, ...]
    // Ordered first by person and then by body part.
    const auto numPeople     = kp.getSize(0);
    const auto numBodyParts  = kp.getSize(1);
    const auto numComponents = kp.getSize(2); // should be 3

    if (numPeople == 0 || numBodyParts == 0 || numComponents < 3) {
        return pose2D; // No detections
    }

    constexpr float kMinJointScore = 0.1f; // TODO: tune this threshold

    // Extract person's joints
    std::vector<Point2D> keypoints(numBodyParts, Point2D());

    for (int part = 0; part < numBodyParts; ++part) {

        // Get keypoint from flattened array
        int baseIndex = 3 * (0 * numBodyParts + part);
        float x       = kp[baseIndex + 0];
        float y       = kp[baseIndex + 1];
        float score   = kp[baseIndex + 2];

        // Zero low confidence joints
        if (score < kMinJointScore) {
            x = 0.0f;
            y = 0.0f;
            score = 0.0f;
        }

        // Store keypoint to structured output
        int outIndex = 3 * part;
        keypoints[part].x = x;
        keypoints[part].y = y;
        keypoints[part].score = score;
    }

    // Return Pose2D.
    // Joints remain in OpenPose BODY_25 order:
    // 0 -> Nose
    // 1 -> Neck
    // 2 -> RShoulder
    // 3 -> RElbow
    // 4 -> RWrist
    // 5 -> LShoulder
    // 6 -> LElbow
    // 7 -> LWrist
    // 8 -> MidHip
    // 9 -> RHip
    // 10 -> RKnee
    // 11 -> RAnkle
    // 12 -> LHip
    // 13 -> LKnee
    // 14 -> LAnkle
    // 15 -> REye
    // 16 -> LEye
    // 17 -> REar
    // 18 -> LEar
    // 19 -> LBigToe
    // 20 -> LSmallToe
    // 21 -> LHeel
    // 22 -> RBigToe
    // 23 -> RSmallToe
    // 24 -> RHeel
    // 25 -> Background

    pose2D.keypoints = keypoints;

    return pose2D;
}