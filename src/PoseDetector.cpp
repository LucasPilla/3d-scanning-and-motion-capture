// PoseDetector.cpp
// Implementation of the PoseDetector class (OpenPose wrapper).

#include "PoseDetector.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

PoseDetector::PoseDetector(std::optional<std::string> precomputedKeypointsPath)
{
    // Only initialize OpenPose if not using precomputed keypoints
    if (precomputedKeypointsPath.has_value()) 
    {
        std::string path = *precomputedKeypointsPath; 
        std::cout << "Using pre-computed keypoints from: " << path << std::endl;
        source_ = PoseSource::Precomputed;
        loadKeypoints(path);
    } 
    else 
    {
        // Configure OpenPose model parameters
        op::WrapperStructPose poseConfig;
        poseConfig.modelFolder = "/opt/openpose/models/";
        poseConfig.renderMode = op::RenderMode::None;
        
        // // Use smaller network input for speed (width, height) in pixels
        // // You can tweak these: smaller = faster but less accurate
        // poseConfig.netInputSize = {960, 544}; // smaller = faster

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
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        std::cerr << "Failed to open keypoints file: " << jsonPath << "\n";
        return false;
    }

    json j;
    in >> j;


    for (auto& [frameStr, joints] : j.items()) {
        std::vector<Point2D> keypoints;
        for (auto& p : joints) {
            keypoints.push_back({
                p["x"].get<float>(),
                p["y"].get<float>(),
                p["score"].get<float>()
            });
        }
        cachedPoses_[std::stoi(frameStr)] = keypoints;
    }

    std::cout << "Loaded keypoints for "
              << cachedPoses_.size() << " frames\n";
    return true;
}

std::vector<Point2D> PoseDetector::detect(const cv::Mat& frame, int frameIdx)
{
    // Precomputed mode
    if (source_ == PoseSource::Precomputed) {
        if (cachedPoses_.count(frameIdx))
            return cachedPoses_[frameIdx];
        return std::vector<Point2D>{};
    }

    // OpenPose live mode
    std::vector<Point2D> keypoints;

    // Convert OpenCV image into OpenPose input format
    auto input = OP_CV2OPCONSTMAT(frame); 

    // Send frame to OpenPose and retrieve output 
    auto result = openpose_->emplaceAndPop(input);

    if (!result || result->empty()) // No detections
        return keypoints;

    const auto& kp = result->at(0)->poseKeypoints;
    int numJoints = kp.getSize(1);

    for (int i = 0; i < numJoints; i++) {
        int idx = 3 * i;
        keypoints.push_back({
            kp[idx],
            kp[idx + 1],
            kp[idx + 2]
        });
    }

    // Return keypoints.
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

    return keypoints;
}