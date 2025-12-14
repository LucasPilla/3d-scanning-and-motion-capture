// main.cpp
// ---------
// Pipeline entry point.
// This only orchestrates the modules; all logic lives in separate classes.
// Steps:
//  1) Load video frames (VideoLoader)
//  2) Extract 2D joints using OpenPose (PoseDetector)
//  3) Fit SMPL to each frame (FittingOptimizer)
//     - baseline: per-frame fitting only
//     - final method: temporal regularization during optimization (see TemporalSmoother)
//  4) Visualize or export results (Visualization)


#include "VideoLoader.h"
#include "PoseDetector.h"
#include "Visualization.h"
#include "FittingOptimizer.h"
#include "TemporalSmoother.h"

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>\n"; // expects a video path as parameter
        return 1;
    }

    std::string videoPath = argv[1];

    // Load video
    VideoLoader loader(videoPath);
    if (!loader.isOpen()) {
        std::cerr << "Error: Cannot open video.\n";
        return 1;
    }

    // Setup OpenPose wrapper
    PoseDetector poseDetector;

    // Setup output video writer
    Visualization visualizer(loader.width(), loader.height(), loader.fps());

    // Placeholder SMPL fitting + temporal smoothing.
    // Configure fitting options (flags).
    FittingOptimizer::Options fitOpts;
    fitOpts.temporalRegularization = false;   // TODO: enable for proposed method
    fitOpts.warmStarting           = true;
    fitOpts.freezeShapeParameters  = false;

    // TODO: Replace nullptr with a real SMPLModel instance once it is implemented.
    FittingOptimizer fitter(nullptr, fitOpts);

    long frameCounter = 0;
    cv::Mat frame;

    while (loader.readFrame(frame)) {

        frameCounter++;

        // During initial development let's work with a small range of frames.
        int startFrame = 1;
        int endFrame   = 100;

        if (frameCounter < startFrame) continue;
        if (frameCounter >= endFrame) break;

        std::cout << "Processing frame " << frameCounter << "\n";

        // Extract pose keypoints
        auto keypoints = poseDetector.detect(frame);

        // Wrap keypoints into Pose2D for fitting.
        Pose2D pose2D;
        pose2D.keypoints = keypoints;
        fitter.fitFrame(pose2D);

        // TODO:
        //  - Collect pose/shape params over all frames.
        //  - After the sequence, pass them through TemporalSmoother.

        // Write output frame
        visualizer.write(frame);  // currently just original frame
    }

    std::cout << "Output written to output.avi\n";
    return 0;
}
