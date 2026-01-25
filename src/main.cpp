// main.cpp
// ---------
// Pipeline entry point.
// This only orchestrates the modules; all logic lives in separate classes.
// Steps:
//  1) Load video frames (VideoLoader)
//  2) Extract 2D joints using OpenPose (PoseDetector)
//  3) Fit SMPL to each frame (SMPLOptimizer)
//  4) Visualize or export results (Visualization)

#include "CameraModel.h"
#include "SMPLOptimizer.h"
#include "PoseDetector.h"
#include "SMPLModel.h"
#include "TemporalSmoother.h"
#include "VideoLoader.h"
#include "Visualization.h"

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>

#include <nlohmann/json.hpp>
using nlohmann::json;

#include <chrono>
using Clock = std::chrono::steady_clock;

struct FrameTimings
{
    double pose_detection_ms = 0.0;
    double fit_rigid_ms      = 0.0;
    double fit_pose_ms       = 0.0;
    double compute_mesh_ms   = 0.0;
    double draw_mesh_ms      = 0.0;
    double total_frame_ms    = 0.0;
};

struct FrameDebugEntry
{
    int frame_index = 0;
    int num_keypoints = 0;

    std::vector<Point2D> keypoints_2d;

    // Optimizer outputs
    Eigen::Matrix3d globalR;
    Eigen::Vector3d globalT;
    std::vector<double> pose_params;
    std::vector<double> shape_params;

    // NEW: timings
    FrameTimings timings;

    std::vector<std::array<double,3>> joints3d;
    std::string mesh_obj_path;

	// Optimization diagnostics
    double fit_rigid_final_cost = -1.0;
    int    fit_rigid_iterations = 0;
    double fit_pose_final_cost  = -1.0;
    int    fit_pose_iterations  = 0;
};

struct PipelineDebugLog
{
    // CLI / general settings
    std::string video_path;
    std::string smpl_path;
    std::string output_folder;
    bool        used_precomputed_keypoints = false;
    std::string precomputed_keypoints_path; // empty if not used
    bool        used_debug_frame = false;
    int         debug_frame_index = -1;
    int         max_frames = -1;

    // Optimizer options
    SMPLOptimizer::Options optimizer_options;

    // Camera / video info
    int    frame_width  = 0;
    int    frame_height = 0;
    double fps          = 0.0;
    double fx           = 0.0;
    double fy           = 0.0;
    double cx           = 0.0;
    double cy           = 0.0;

    // Per-frame entries
    std::vector<FrameDebugEntry> frames;
};

inline void to_json(json &j, const SMPLOptimizer::Options &opt)
{
    j = json{
        {"temporalRegularization", opt.temporalRegularization},
        {"warmStarting", opt.warmStarting},
        {"freezeShapeParameters", opt.freezeShapeParameters},
    };
}

inline void to_json(json &j, const FrameDebugEntry &f)
{
    j = json{};
    j["frame_index"]   = f.frame_index;
    j["num_keypoints"] = f.num_keypoints;

    // 2D keypoints
    json kps = json::array();
    for (const auto &kp : f.keypoints_2d)
    {
        kps.push_back(json{
            {"x", kp.x},
            {"y", kp.y},
            {"score", kp.score},
        });
    }
    j["keypoints_2d"] = std::move(kps);

    // Global rotation (3x3) as nested array
    json R = json::array();
    for (int i = 0; i < 3; ++i)
    {
        json row = json::array();
        for (int jcol = 0; jcol < 3; ++jcol)
        {
            row.push_back(f.globalR(i, jcol));
        }
        R.push_back(std::move(row));
    }
    j["globalR"] = std::move(R);

    // Global translation (3)
    json T = json::array();
    T.push_back(f.globalT(0));
    T.push_back(f.globalT(1));
    T.push_back(f.globalT(2));
    j["globalT"] = std::move(T);

    // Pose and shape params as flat arrays
    j["pose_params"]  = f.pose_params;
    j["shape_params"] = f.shape_params;

	// Timings
	json jt;
	jt["pose_detection_ms"] = f.timings.pose_detection_ms;
	jt["fit_rigid_ms"]      = f.timings.fit_rigid_ms;
	jt["fit_pose_ms"]       = f.timings.fit_pose_ms;
	jt["compute_mesh_ms"]   = f.timings.compute_mesh_ms;
	jt["draw_mesh_ms"]      = f.timings.draw_mesh_ms;
	jt["total_frame_ms"]    = f.timings.total_frame_ms;
	j["timings"] = std::move(jt);

	// 3D joints
	json joints = json::array();
	for (const auto &p : f.joints3d)
	{
		joints.push_back(json::array({p[0], p[1], p[2]}));
	}
	j["joints3d"] = std::move(joints);

	// Mesh path
	j["mesh_obj_path"] = f.mesh_obj_path;

	// Optimization diagnostics
	json jc;
	jc["fit_rigid_final_cost"] = f.fit_rigid_final_cost;
	jc["fit_rigid_iterations"] = f.fit_rigid_iterations;
	jc["fit_pose_final_cost"]  = f.fit_pose_final_cost;
	jc["fit_pose_iterations"]  = f.fit_pose_iterations;
	j["optimizer_costs"] = std::move(jc);
}

inline void to_json(json &j, const PipelineDebugLog &p)
{
    j = json{};  // start with an empty object

    // CLI / general
    j["video_path"]  = p.video_path;
    j["smpl_path"]   = p.smpl_path;
    j["output_folder"] = p.output_folder;
    j["used_precomputed_keypoints"] = p.used_precomputed_keypoints;
    j["precomputed_keypoints_path"] = p.precomputed_keypoints_path;
    j["used_debug_frame"]           = p.used_debug_frame;
    j["debug_frame_index"]          = p.debug_frame_index;
    j["max_frames"]                 = p.max_frames;

    // Optimizer
    j["optimizer_options"] = p.optimizer_options;

    // Camera / video
    j["frame_width"]  = p.frame_width;
    j["frame_height"] = p.frame_height;
    j["fps"]          = p.fps;
    j["fx"]           = p.fx;
    j["fy"]           = p.fy;
    j["cx"]           = p.cx;
    j["cy"]           = p.cy;

    // Per-frame
    j["frames"] = p.frames;
}

int main(int argc, char *argv[])
{

	argparse::ArgumentParser program("pipeline");

	// Pipeline requires a video path
	program.add_argument("--video-path").help("Path to video file").required();

	// Pipeline requires a SMPL model path
	program.add_argument("--smpl-path")
		.help("Path to model generated by preprocess.py (.json)")
		.required();

	// Optional path to output folder
	// Default is "./output"
	program.add_argument("--output")
		.help("Output folder to save results.")
		.default_value("./output");

	// Optional path to pre-computed keypoints
	// If not defined, OpenPose will run for each frame
	program.add_argument("--precomputed-keypoints")
		.help("Path to pre-computed keypoints (.json)");

	// Optional: process only a single frame and save debug images
	program.add_argument("--debug-frame")
		.help(
			"If set, process only this 1-based frame index and save debug images")
		.scan<'i', int>();

	// Optional flag to enable debug logging
	program.add_argument("--debug-log")
		.help("If set, save per-frame debug data (JSON, meshes, etc.)")
		.default_value(false)
		.implicit_value(true);

	try
	{
		// Parse args
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error &err)
	{
		// This block runs if the user forgets the argument
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	// Load command line arguments into variables
	std::string videoPath = program.get("--video-path");

	std::string smplPath = program.get("--smpl-path");

	std::filesystem::path outputFolder = program.get("--output");

	std::optional<std::string> precomputedKeypointsPath = std::nullopt;
	if (program.is_used("--precomputed-keypoints"))
		precomputedKeypointsPath = program.get("--precomputed-keypoints");

	std::optional<int> debugFrame = std::nullopt;
	if (program.is_used("--debug-frame"))
		debugFrame = program.get<int>("--debug-frame");

	bool debugLogEnabled = program.get<bool>("--debug-log");

	// Create output folder
	std::filesystem::create_directory(outputFolder);

	// Initialize video loader
	VideoLoader loader(videoPath);

	// Initialize pose detector
	PoseDetector poseDetector(precomputedKeypointsPath);

	// Initialize output video writer
	Visualization visualizer(loader.width(), loader.height(), loader.fps());

	// Initialize simple pinhole camera intrinsics (approximate)
	double fx = static_cast<double>(loader.width());
	double fy = static_cast<double>(loader.width());
	double cx = static_cast<double>(loader.width()) / 2.0f;
	double cy = static_cast<double>(loader.height()) / 2.0f;
	CameraModel cameraModel(fx, fy, cx, cy);

	// Load SMPL model (preprocessed JSON).
	SMPLModel smplModel;
	if (!smplModel.loadFromJson(smplPath))
	{
		std::cerr << "Warning: Failed to load SMPL model from " << smplPath
				  << std::endl;
	}

	// Configure optimizer fitting options (flags).
	SMPLOptimizer::Options fitOpts;
	fitOpts.temporalRegularization = false;
	fitOpts.warmStarting = true;
	fitOpts.freezeShapeParameters = false;

	const int maxFrames = 100;

	// Minimal debug log container
	PipelineDebugLog debugLog;
	if (debugLogEnabled)
	{
		// CLI / general
		debugLog.video_path  = videoPath;
		debugLog.smpl_path   = smplPath;
		debugLog.output_folder = outputFolder.string();
	
		debugLog.used_precomputed_keypoints = precomputedKeypointsPath.has_value();
		debugLog.precomputed_keypoints_path = precomputedKeypointsPath.value_or("");
	
		debugLog.used_debug_frame  = debugFrame.has_value();
		debugLog.debug_frame_index = debugFrame.value_or(-1);
	
		debugLog.max_frames = maxFrames;
	
		// Optimizer
		debugLog.optimizer_options = fitOpts;
	
		// Camera / video
		debugLog.frame_width  = loader.width();
		debugLog.frame_height = loader.height();
		debugLog.fps          = loader.fps();
	
		debugLog.fx = fx;
		debugLog.fy = fy;
		debugLog.cx = cx;
		debugLog.cy = cy;
	}

	// Initialize optimizer using SMPLModel instance
	SMPLOptimizer fitter(&smplModel, &cameraModel, fitOpts);

	if (debugLogEnabled)
	{
		std::cout << "[DEBUG] Debug logging is ENABLED for this run." << std::endl;
	}

	int frameIdx = 0;
	cv::Mat frame;

	while (loader.readFrame(frame))
	{
		auto tFrameStart = Clock::now();

		frameIdx++;

		// debug-frame handling
		if (debugFrame.has_value())
		{
			if (frameIdx < *debugFrame)
				continue;
			if (frameIdx > *debugFrame)
				break;
		}

		if (frameIdx > maxFrames)
			break;

		cv::Mat frameInput = frame.clone();

		// Pose detection
		auto tDetStart = Clock::now();
		Pose2D pose2D = poseDetector.detect(frame, frameIdx);
		auto tDetEnd  = Clock::now();

		// Optimization
		auto tOptStart = Clock::now();
		fitter.fitFrame(pose2D);
		auto tOptEnd   = Clock::now();

		// Mesh computation
		auto tMeshStart = Clock::now();
		SMPLMesh mesh = smplModel.computeMesh();

		std::filesystem::path objPath;
		if (debugLogEnabled)
		{
			std::ostringstream oss;
			oss << "frame_" << std::setw(4) << std::setfill('0') << frameIdx << "_mesh.obj";
			objPath = outputFolder / oss.str();
			mesh.save(objPath.string());
		}
		auto tMeshEnd = Clock::now();

		// Draw + write
		auto tDrawStart = Clock::now();
		cv::Mat frameMesh = frameInput.clone();
		visualizer.drawMesh(frameMesh, mesh, cameraModel,
							fitter.getGlobalR(), fitter.getGlobalT(),
							cv::Scalar(0, 255, 255), 1);
		visualizer.write(frameMesh);
		auto tDrawEnd = Clock::now();

		auto tFrameEnd = Clock::now();

		auto toMs = [](auto dt) {
			return std::chrono::duration_cast<
					std::chrono::duration<double, std::milli>>(dt)
				.count();
		};

		double detectMs = toMs(tDetEnd - tDetStart);
		double optMs    = toMs(tOptEnd - tOptStart);
		double meshMs   = toMs(tMeshEnd - tMeshStart);
		double drawMs   = toMs(tDrawEnd - tDrawStart);
		double totalMs  = toMs(tFrameEnd - tFrameStart);

		// Console output
		std::cout << "Frame " << frameIdx
				<< " | detect: " << detectMs << " ms"
				<< " | opt: "    << optMs    << " ms"
				<< " | mesh: "   << meshMs   << " ms"
				<< " | draw: "   << drawMs   << " ms"
				<< " | total: "  << totalMs  << " ms"
				<< std::endl;

		if (debugLogEnabled)
		{
			FrameDebugEntry entry;
			entry.frame_index   = frameIdx;
			entry.num_keypoints = static_cast<int>(pose2D.keypoints.size());
			entry.keypoints_2d  = pose2D.keypoints;

			entry.globalR      = fitter.getGlobalR();
			entry.globalT      = fitter.getGlobalT();
			entry.pose_params  = fitter.getPoseParams();
			entry.shape_params = fitter.getShapeParams();

			entry.timings.pose_detection_ms = detectMs;
			entry.timings.fit_rigid_ms      = 0.0;
			entry.timings.fit_pose_ms       = optMs;
			entry.timings.compute_mesh_ms   = meshMs;
			entry.timings.draw_mesh_ms      = drawMs;
			entry.timings.total_frame_ms    = totalMs;

			// Optimization diagnostics
			entry.fit_rigid_final_cost = fitter.getLastFitRigidCost();
			entry.fit_rigid_iterations = fitter.getLastFitRigidIters();
			entry.fit_pose_final_cost  = fitter.getLastFitPoseCost();
			entry.fit_pose_iterations  = fitter.getLastFitPoseIters();

			// 3D joints
			Eigen::Matrix<double, 24, 3> J3D = smplModel.getLastJoints3D();
			entry.joints3d.reserve(24);
			for (int jIdx = 0; jIdx < 24; ++jIdx)
			{
				entry.joints3d.push_back({J3D(jIdx, 0), J3D(jIdx, 1), J3D(jIdx, 2)});
			}

			// Mesh path
			if (!objPath.empty())
				entry.mesh_obj_path = objPath.string();

			debugLog.frames.push_back(std::move(entry));
		}

		// debug-frame PNG block as you already have…
	}

	if (debugLogEnabled)
	{
		std::filesystem::path logPath = outputFolder / "pipeline_debug.json";
		json j = debugLog;

		std::ofstream out(logPath);
		if (out.is_open())
		{
			out << j.dump(2); // pretty-print
			std::cout << "[DEBUG] Saved debug log to " << logPath << std::endl;
		}
		else
		{
			std::cerr << "[DEBUG] Failed to open debug log file at " << logPath << std::endl;
		}
	}

	return 0;
}