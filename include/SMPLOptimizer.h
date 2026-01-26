// SMPLOptimizer
// -----------------
// Performs non-linear optimization (SMPLify-style) to fit SMPL parameters
// to 2D OpenPose detections using Ceres Solver.

#pragma once

#include "CameraModel.h"
#include "PoseDetector.h"
#include "SMPLModel.h"
#include "TemporalSmoother.h"
#include <Eigen/Dense>
#include <vector>

// Interface for SMPL fitting
class SMPLOptimizer
{
public:
	// Configuration flags controlling advanced features
	// These correspond to the project proposal:
	//  - TEMPORAL_REGULARIZATION
	//  - WARM_STARTING
	//  - FREEZE_SHAPE_PARAMETERS
	struct Options
	{
		bool temporalRegularization = false;
		bool warmStarting = false;
		bool freezeShapeParameters = false;
	};

	explicit SMPLOptimizer(SMPLModel *smplModel, CameraModel *cameraModel,
							  const Options &options);

	// Fit SMPL parameters to a single frame
	void fitFrame(const Pose2D &observation);

	// Step 1: Fit a global 3D rigid transform (R, t) in camera space so that
	// SMPL 3D joints align better with OpenPose 2D detections
	void fitRigid(const Pose2D &observation);

	// Step 2: Optimize SMPL pose parameters using reprojection error
	void fitPose(const Pose2D &observation);

	// Get the global rigid transform computed in Step 1
	const Eigen::Matrix3d &getGlobalR() const { return globalR_; }
	const Eigen::Vector3d &getGlobalT() const { return globalT_; }

	// Expose current SMPL pose and shape parameters
	const std::vector<double> &getPoseParams() const { return poseParams; }
	const std::vector<double> &getShapeParams() const { return shapeParams; }

	// Expose last optimization diagnostics
	double getLastFitRigidCost() const { return lastFitRigidCost_; }
	int    getLastFitRigidIters() const { return lastFitRigidIters_; }
	double getLastFitPoseCost() const { return lastFitPoseCost_; }
	int    getLastFitPoseIters() const { return lastFitPoseIters_; }

private:
	// Global rigid transform computed in Step 1 (fitRigid)
	Eigen::Matrix3d globalR_ = Eigen::Matrix3d::Identity();
	Eigen::Vector3d globalT_ = Eigen::Vector3d::Zero();

	// Configuration flags (see Options above).
	Options options;

	// SMPL pose parameters (e.g., 72-dim axis-angle)
	std::vector<double> poseParams;

	// SMPL shape parameters (e.g., 10 betas)
	std::vector<double> shapeParams;

	// 2D joints for the current frame
	Pose2D current2DJoints;

	// Pointer to SMPL model
	SMPLModel *smplModel = nullptr;

	// Pointer to camera model
	CameraModel *cameraModel = nullptr;

	// History of parameters for temporal smoothing / regularization
	TemporalSmoother smoother;
	TemporalSmoother::ParamSequence poseHistory;
	TemporalSmoother::ParamSequence shapeHistory;

	// Tracks whether we already have a solution from a previous frame
    bool hasPreviousFrame_ = false;

	// Ceres costs and iterations
	double lastFitRigidCost_ = -1.0;
	int    lastFitRigidIters_ = 0;
	double lastFitPoseCost_ = -1.0;
	int    lastFitPoseIters_ = 0;
};