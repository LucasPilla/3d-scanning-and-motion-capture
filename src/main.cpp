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
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>\n"; // expects a video path as parameter
        return 1;
    }

    std::string videoPath = argv[1];

    // Load video
    VideoLoader loader(videoPath);

    // Setup OpenPose wrapper
    PoseDetector poseDetector;

    // Setup output video writer
    Visualization visualizer(loader.width(), loader.height(), loader.fps());

    // Load SMPL model (preprocessed JSON).
    // NOTE: Adjust this path to wherever your teammate writes the JSON,
    // e.g. "models/smpl_male.json" created by preprocess.py.
    SMPLModel smplModel;
    const std::string smplJsonPath = "models/smpl_male.json"; // TODO: confirm path with team

    if (!smplModel.loadFromJson(smplJsonPath)) {
        std::cerr << "Warning: Failed to load SMPL model from " << smplJsonPath
                << ". Fitting will not use a real model yet.\n";
    }

    /*

    //Will be used to test after we load the model
    // 1 - SMPL sanity test ------------------------------------------
    std::vector<double> zeroPose(72, 0.0);
    std::vector<double> zeroShape(10, 0.0);

    smplModel.setPose(zeroPose);
    smplModel.setShape(zeroShape);

    SMPLMesh testMesh = smplModel.getMesh();

    std::cout << "SMPL test mesh: "
            << testMesh.vertices.size() << " vertices, "
            << testMesh.faces.size() << " faces\n";

    //Expected output:
    //SMPLModel::loadFromJson - loaded model
    //SMPL test mesh: 6890 vertices, 13776 faces
    //----------------------------------------------------------    

    //Also more sanity checks later
    */

    // Placeholder SMPL fitting + temporal smoothing.
    // Configure fitting options (flags).
    FittingOptimizer::Options fitOpts;
    fitOpts.temporalRegularization = false;
    fitOpts.warmStarting           = false;
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

        // Extract pose
        Pose2D pose2D = poseDetector.detect(frame);

        // Run optimizer
        // The proposed enhacements for temporal consistency are applied within the optimizer.
        fitter.fitFrame(pose2D);

        // Write output frame
        visualizer.drawKeypoints(frame, pose2D.keypoints);
        visualizer.write(frame);  // currently just original frame
    }

    std::cout << "Output written to output.mp4\n";
    return 0;
}
