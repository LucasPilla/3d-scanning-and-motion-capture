// extract_keypoints.cpp
// 
// Standalone preprocessing tool.
// Runs OpenPose ONCE on a video and saves 2D keypoints
// to a JSON file for fast reuse in later experiments.
//
// Usage:
//   ./extract_keypoints video.mp4 keypoints.json
//
// This avoids running OpenPose during SMPL fitting,
// making development and debugging much faster.


#include <openpose/headers.hpp>     // OpenPose API
#include <opencv2/opencv.hpp>       // Video loading
#include <nlohmann/json.hpp>        // JSON serialization

#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
    // Expects two arguments: input video path, output JSON file path
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <video_path> <output_json>\n";
        return 1;
    }

    std::string videoPath = argv[1];
    std::string outputPath = argv[2];

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << videoPath << "\n";
        return 1;
    }

    // ---------------- Configure OpenPose ----------------
    // We disable rendering because we only need keypoints
    op::WrapperStructPose poseConfig;
    poseConfig.modelFolder = "/opt/openpose/models/";
    poseConfig.renderMode  = op::RenderMode::None;

    // Create OpenPose wrapper
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    opWrapper.configure(poseConfig);
    opWrapper.start();

    // JSON object storing all frames
    nlohmann::json allFrames;

    int frameIdx = 0;
    cv::Mat frame;

    // Process video frame-by-frame
    while (cap.read(frame)) {
        frameIdx++;

        // Convert OpenCV image to OpenPose format
        auto input = OP_CV2OPCONSTMAT(frame);

        // Run OpenPose inference
        auto datum = opWrapper.emplaceAndPop(input);

        // Skip frames with no detection
        if (!datum || datum->empty()) {
            continue;
        }

        // Access detected pose keypoints
        const auto& keypoints = datum->at(0)->poseKeypoints;

        // Only store the most confident detected person
        if (keypoints.getSize(0) > 0) {

            nlohmann::json joints;
            int numParts = keypoints.getSize(1);

            // Iterate over body joints
            for (int j = 0; j < numParts; j++) {
                int idx = 3 * j;

                joints.push_back({
                    {"x",     keypoints[idx]},
                    {"y",     keypoints[idx + 1]},
                    {"score", keypoints[idx + 2]} //confidence score
                });
            }

            // Store joints under frame index
            allFrames[std::to_string(frameIdx)] = joints;
        }

        std::cout << "Processed frame " << frameIdx << "\n";
    }

    // Save JSON to disk 
    std::ofstream out(outputPath);
    out << allFrames.dump(2);  // Pretty-printed JSON
    out.close();

    std::cout << "Saved keypoints to " << outputPath << "\n";
    return 0;
}
