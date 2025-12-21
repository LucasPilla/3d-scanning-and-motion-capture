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
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

PoseDetector::PoseDetector(PoseSource source)
    : source_(source)
{
    // Only initialize OpenPose if needed
    if (source_ == PoseSource::OpenPoseLive) {

        // Configure OpenPose model parameters
        op::WrapperStructPose poseConfig;
        poseConfig.modelFolder = "/opt/openpose/models/";
        poseConfig.renderMode = op::RenderMode::None;
        
        // Use smaller network input for speed (width, height) in pixels
        // You can tweak these: smaller = faster but less accurate
        poseConfig.netInputSize = {320, 176}; // smaller = faster

        // Disable rendering overlays to save CPU (we don't use the rendered image)
        poseConfig.renderMode = op::RenderMode::None;

        openpose_ = std::make_unique<op::Wrapper>(
            op::ThreadManagerMode::Asynchronous
        );
        openpose_->configure(poseConfig);
        // Start the OpenPose processing threads. Must be called before detecting any frames
        openpose_->start();
    }
}

bool PoseDetector::loadKeypoints(const std::string& jsonPath)
{
    if (source_ != PoseSource::Precomputed) {
        std::cerr << "loadKeypoints called in non-precomputed mode\n";
        return false;
    }

    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        std::cerr << "Failed to open keypoints file: " << jsonPath << "\n";
        return false;
    }

    json j;
    in >> j;


    for (auto& [frameStr, joints] : j.items()) {
        Pose2D pose;
        for (auto& p : joints) {
            pose.keypoints.push_back({
                p["x"].get<float>(),
                p["y"].get<float>(),
                p["score"].get<float>()
            });
        }
        cachedPoses_[std::stoi(frameStr)] = pose;
    }

    std::cout << "Loaded keypoints for "
              << cachedPoses_.size() << " frames\n";
    return true;
}

Pose2D PoseDetector::detect(const cv::Mat& frame, int frameIdx)
{
    //Precomputed mode
    if (source_ == PoseSource::Precomputed) {
        if (cachedPoses_.count(frameIdx))
            return cachedPoses_[frameIdx];
        return Pose2D{};
    }

    // OpenPose live mode
    Pose2D pose2D;
    auto input = OP_CV2OPCONSTMAT(frame); // Convert OpenCV image into OpenPose input format
    auto result = openpose_->emplaceAndPop(input); // Send frame to OpenPose and retrieve output 

    if (!result || result->empty()) // No detections
        return pose2D;

    const auto& kp = result->at(0)->poseKeypoints;
    int numJoints = kp.getSize(1);

    for (int i = 0; i < numJoints; i++) {
        int idx = 3 * i;
        pose2D.keypoints.push_back({
            kp[idx],
            kp[idx + 1],
            kp[idx + 2]
        });
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

    return pose2D;
}