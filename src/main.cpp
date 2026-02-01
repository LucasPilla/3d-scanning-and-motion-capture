// main.cpp
// ---------
// Pipeline entry point for SMPL body model fitting.

#include "CameraModel.h"
#include "PoseDetector.h"
#include "SMPLModel.h"
#include "SMPLOptimizer.h"
#include "VideoLoader.h"
#include "Visualization.h"

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct DebugData {
    json config;                    // Pipeline configuration
    std::vector<json> frames;       // Per-frame data
    
    void save(const fs::path& path) const {
        json output;
        output["config"] = config;
        output["frames"] = frames;
        
        std::ofstream file(path);
        if (file.is_open()) {
            file << output.dump(2);
            std::cout << "[DEBUG] Saved to " << path << "\n";
        } else {
            std::cerr << "[DEBUG] Failed to save to " << path << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    // Parse command line arguments
    argparse::ArgumentParser program("pipeline");

    program.add_argument("--video-path")
        .help("Path to video file")
        .required();

    program.add_argument("--smpl-path")
        .help("Path to SMPL model (.json)")
        .required();

    program.add_argument("--output")
        .help("Output folder")
        .default_value("./output");

    program.add_argument("--precomputed-keypoints")
        .help("Path to pre-computed keypoints (.json)");

    program.add_argument("--frame")
        .help("Process only this frame")
        .scan<'i', int>();

    program.add_argument("--debug")
        .help("Save debug data as a JSON")
        .default_value(false)
        .implicit_value(true);
    
    program.add_argument("--skip-viz")
        .help("Skip visualization")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    std::string videoPath = program.get("--video-path");
    std::string smplPath = program.get("--smpl-path");
    fs::path outputFolder = program.get("--output");
    bool debugMode = program.get<bool>("--debug");
    bool skipViz = program.get<bool>("--skip-viz");

    std::optional<std::string> precomputedKeypoints;
    if (program.is_used("--precomputed-keypoints"))
        precomputedKeypoints = program.get("--precomputed-keypoints");

    std::optional<int> specificFrame;
    if (program.is_used("--frame"))
        specificFrame = program.get<int>("--frame");

    // Create directories for output
    fs::create_directory(outputFolder);
    fs::create_directory(outputFolder / "meshes");

    // Initialize video reader
    VideoLoader loader(videoPath);

    // Initialize OpenPose detector
    PoseDetector poseDetector(precomputedKeypoints);

    // Initialize visualizer
    std::filesystem::path outputVideoPath = outputFolder / "output.mp4";
    Visualization visualizer(loader.width(), loader.height(), loader.fps(), outputVideoPath);

    // Initialize camera model
    CameraModel camera(loader.width(), loader.height());

    // Initialize SMPL model
    SMPLModel smplModel;
    if (!smplModel.loadFromJson(smplPath)) {
        std::cerr << "Failed to load SMPL model from " << smplPath << "\n";
        return 1;
    }

    // Initializer optimizer
    SMPLOptimizer::Options fitOpts;
    fitOpts.temporalRegularization = true;
    fitOpts.warmStarting = true;
    fitOpts.freezeShapeParameters = true;
    SMPLOptimizer optimizer(&smplModel, &camera, fitOpts);

    DebugData debug;
    
    debug.config = {
        {"video_path", videoPath},
        {"smpl_path", smplPath},
        {"output_folder", outputFolder.string()},
        {"frame_width", loader.width()},
        {"frame_height", loader.height()},
        {"fps", loader.fps()},
        {"camera", {
            {"fx", camera.intrinsics().fx}, 
            {"fy", camera.intrinsics().fy}, 
            {"cx", camera.intrinsics().cx}, 
            {"cy", camera.intrinsics().cy}
        }},
        {"optimizer", {
            {"temporal_regularization", fitOpts.temporalRegularization},
            {"warm_starting", fitOpts.warmStarting},
            {"freeze_shape", fitOpts.freezeShapeParameters}
        }}
    };

    int frameIdx = 0;
    cv::Mat frame;

    while (loader.readFrame(frame)) {
        frameIdx++;

        if (frameIdx > 600) break;

        // Handle debug-frame mode
        if (specificFrame) {
            if (frameIdx < *specificFrame) continue;
            if (frameIdx > *specificFrame) break;
        }

        auto tStart = Clock::now();

        // Pose Detection
        auto t0 = Clock::now();
        auto keypoints = poseDetector.detect(frame, frameIdx);
        double detectMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // Optimization
        t0 = Clock::now();
        optimizer.fitFrame(keypoints);
        double optMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // Mesh Computation
        t0 = Clock::now();
        SMPLMesh mesh = smplModel.computeMesh();
        double meshMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // Visualization
        t0 = Clock::now();
        if (!skipViz) {
            cv::Mat outputFrame = frame.clone();
            visualizer.drawKeypoints(outputFrame, keypoints);
            visualizer.drawMesh(outputFrame, mesh, camera, optimizer.getGlobalT());
            visualizer.write(outputFrame);
        }
        double drawMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        double totalMs = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count();

        // Console output
        std::cout << "Frame " << frameIdx
                  << " | detect: " << detectMs << " ms"
                  << " | opt: " << optMs << " ms"
                  << " | mesh: " << meshMs << " ms"
                  << " | draw: " << drawMs << " ms"
                  << " | total: " << totalMs << " ms\n";

        // Collect debug data
        if (debugMode) {
            json kps = json::array();
            for (const auto& kp : keypoints) {
                kps.push_back({{"x", kp.x}, {"y", kp.y}, {"score", kp.score}});
            }

            Eigen::Vector3d T = optimizer.getGlobalT();

            // Save mesh
            std::stringstream meshFilename;
            meshFilename << "mesh_" << std::setw(5) << std::setfill('0') << frameIdx << ".obj";
            std::filesystem::path meshPath = outputFolder / "meshes" / meshFilename.str();
            mesh.save(meshPath);
            
            debug.frames.push_back({
                {"frame", frameIdx},
                {"keypoints", kps},
                {"globalT", {T(0), T(1), T(2)}},
                {"pose_params", optimizer.getPoseParams()},
                {"shape_params", optimizer.getShapeParams()},
                {"mesh_path", meshPath.string()},
                {"timings", {
                    {"detect_ms", detectMs},
                    {"opt_ms", optMs},
                    {"mesh_ms", meshMs},
                    {"draw_ms", drawMs},
                    {"total_ms", totalMs}
                }},
                {"optimization", {
                    {"init_cost", optimizer.getInitSummary().final_cost},
                    {"init_iterations", optimizer.getInitSummary().num_successful_steps},
                    {"full_cost", optimizer.getFullSummary().final_cost},
                    {"full_iterations", optimizer.getFullSummary().num_successful_steps}
                }}
            });
        }
    }

    // Save debug data
    if (debugMode) {
        debug.save(outputFolder / "debug.json");
    }

    return 0;
}
