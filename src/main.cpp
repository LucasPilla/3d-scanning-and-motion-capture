// main.cpp
// ---------
// Pipeline entry point.
// This only orchestrates the modules; all logic lives in separate classes.
// Steps:
//  1) Load video frames (VideoLoader)
//  2) Extract 2D joints using OpenPose (PoseDetector)
//  3) Fit SMPL to each frame (FittingOptimizer)
//  4) Visualize or export results (Visualization)

#include "VideoLoader.h"
#include "PoseDetector.h"
#include "Visualization.h"
#include "FittingOptimizer.h"
#include "TemporalSmoother.h"
#include "SMPLModel.h"

#include <iostream>

int main(int argc, char* argv[])
{
    // OpenPose mode. Toggle here.
    // True if precomputed keypoints to be used. False if live OpenPose detection to be used.
    bool usePrecomputed = true;

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>\n"; // expects a video path as parameter
        return 1;
    }

    std::string videoPath = argv[1];

    // Load video
    VideoLoader loader(videoPath);

    // Setup OpenPose wrapper
    PoseDetector poseDetector(
        usePrecomputed ? PoseSource::Precomputed
                       : PoseSource::OpenPoseLive
    );

    if (usePrecomputed) {
        poseDetector.loadKeypoints("keypoints.json");
    }

    // Setup output video writer
    Visualization visualizer(loader.width(), loader.height(), loader.fps());

    // Load SMPL model (preprocessed JSON).
    // "models/smpl_male.json" is created by scripts/preprocess.py.
    SMPLModel smplModel;
    const std::string smplJsonPath = "models/smpl_male.json";

    if (!smplModel.loadFromJson(smplJsonPath)) {
        std::cerr << "Warning: Failed to load SMPL model from " << smplJsonPath
                  << ". Fitting will not use a real model yet.\n";
    }

    // Set dummy SMPL parameters for now: zero pose (72) and zero shape (10)
    if (smplModel.isLoaded()) {
        std::vector<double> dummyPose(72, 0.0);
        std::vector<double> dummyShape(10, 0.0);
        smplModel.setPose(dummyPose);
        smplModel.setShape(dummyShape);

        // Optional one-time sanity check
        SMPLMesh testMesh = smplModel.getMesh();
        std::cout << "Initial SMPL mesh: "
                  << testMesh.vertices.size() << " vertices, "
                  << testMesh.faces.size()    << " faces\n";
    }

    // Placeholder SMPL fitting + temporal smoothing.
    // Configure fitting options (flags).
    FittingOptimizer::Options fitOpts;
    fitOpts.temporalRegularization = false;
    fitOpts.warmStarting           = false;
    fitOpts.freezeShapeParameters  = false;

    // Use a real SMPLModel instance so the optimizer can access it later
    FittingOptimizer fitter(&smplModel, fitOpts);

    int frameIdx = 0;
    cv::Mat frame;

    while (loader.readFrame(frame)) {

        frameIdx++;

        // During initial development you can clamp the frame range if needed.
        std::cout << "Processing frame " << frameIdx << "\n";

        // Extract pose
        Pose2D pose2D = poseDetector.detect(frame, frameIdx);

        // Run optimizer (currently a stub, just prepares data)
        // The proposed enhancements for temporal consistency are applied within the optimizer.
        fitter.fitFrame(pose2D);

        // Trigger SMPL forward pass once per frame with the current (dummy) params
        if (smplModel.isLoaded()) {
            SMPLMesh frameMesh = smplModel.getMesh();

            // Only print once to avoid spamming
            if (frameIdx == 1) {
                std::cout << "Per-frame SMPL mesh: "
                          << frameMesh.vertices.size() << " vertices, "
                          << frameMesh.faces.size()    << " faces\n";
            }

            // TODO (later): use frameMesh for visualization or analysis
        }

        // Write output frame
        visualizer.drawKeypoints(frame, pose2D.keypoints);
        visualizer.write(frame);  // currently just original frame with 2D skeleton overlay
    }

    std::cout << "Output written to output.mp4\n";
    return 0;
}