// main.cpp
// ---------
// Pipeline entry point.
// This only orchestrates the modules; all logic lives in separate classes.
// Steps:
//  1) Load video frames (VideoLoader)
//  2) Extract 2D joints using OpenPose (PoseDetector)
//  3) Fit SMPL to each frame (FittingOptimizer)
//  4) Apply temporal smoothing (TemporalSmoother)
//  5) Visualize or export results (Visualization)

//TODO : REMOVE THIS COMMENTED CODE IF EVERYTING IS WORKING RIGHT
/***
#include <openpose/headers.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    // This script expects a video path as parameter
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }

    std::string videoPath = argv[1];

    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video: " << videoPath << std::endl;
        return -1;
    }

    // Setup video writer
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(
        "output.avi", 
        cv::VideoWriter::fourcc('M','J','P','G'), 
        fps, 
        cv::Size(width, height)
    );

    // Configure OpenPose
    op::WrapperStructPose poseConfig{};
    
    poseConfig.modelFolder = "/opt/openpose/models/"; 
    poseConfig.renderMode = op::RenderMode::Cpu; 

    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    opWrapper.configure(poseConfig);

    // Start OpenPose
    opWrapper.start();

    long frameCounter = 0;

    cv::Mat frame;

    while (true) {

        cap >> frame;
        if (frame.empty()) break;

        // Increment counter
        frameCounter++;

        // During initial development let's work with a small range of frames.
        int startFrame = 800;
        int endFrame = 801;
        if (frameCounter < startFrame) continue;
        if (frameCounter >= endFrame) break;
        std::cout << "Processing frame " << frameCounter << std::endl;

        // Convert to OpenPose format
        auto frameToProcess = OP_CV2OPCONSTMAT(frame);

        // Process frame
        auto opDatum = opWrapper.emplaceAndPop(frameToProcess);
        if (opDatum != nullptr && !opDatum->empty()) {

            // The poseKeypoints attribute is an array with shape [Num People, Body Parts, 3]
            // The last dimention consists of (x, y, score) values
            const auto& poseKeypoints = opDatum->at(0)->poseKeypoints;
            int numberPeople = poseKeypoints.getSize(0);
            int numberBodyParts = poseKeypoints.getSize(1);

            // This demonstrates how to access OpenPose results
            for (int person = 0; person < numberPeople; person++) {
                for (int part = 0; part < numberBodyParts; part++) {
                    // Calculate index in the flattened array
                    // Format: [x, y, score, x, y, score...]
                    int baseIndex = 3 * (person * numberBodyParts + part);
                    float x = poseKeypoints[baseIndex];
                    float y = poseKeypoints[baseIndex + 1];
                    float score = poseKeypoints[baseIndex + 2];
                    std::cout << x << " " << y << " " << score << std::endl;
                }
            }

            // Get output frame
            const auto& opFrame = opDatum->at(0)->cvOutputData;
            cv::Mat outputFrame = OP_OP2CVCONSTMAT(opFrame);

            // Write to video
            writer.write(outputFrame);
        }
    }

    cap.release();
    writer.release();
    opWrapper.stop();
    
    std::cout << "Output video saved to './build/output.avi'" << std::endl;

    return 0;
}

*///


#include "VideoLoader.h"
#include "PoseDetector.h"
#include "Visualization.h"

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

    long frameCounter = 0;
    cv::Mat frame;

    while (loader.readFrame(frame)) {

        frameCounter++;

        // During initial development let's work with a small range of frames.
        int startFrame = 800;
        int endFrame   = 801;

        if (frameCounter < startFrame) continue;
        if (frameCounter >= endFrame) break;

        std::cout << "Processing frame " << frameCounter << "\n";

        // Extract pose keypoints
        auto keypoints = poseDetector.detect(frame);

        // (Later: store keypoints → fit SMPL → smooth)
        // For now: pure debugging output

        // Write output frame
        visualizer.write(frame);  // currently just original frame
    }

    std::cout << "Output written to output.avi\n";
    return 0;
}
