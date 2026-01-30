// SMPLOptimizer
// -----------------
// Performs non-linear optimization (SMPLify-style) to fit SMPL parameters
// to 2D OpenPose detections using Ceres Solver.

#pragma once

#include "CameraModel.h"
#include "PoseDetector.h"
#include "SMPLModel.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

// Interface for SMPL fitting
class SMPLOptimizer {
public:

  // These correspond to the project proposal:
  //  - TEMPORAL_REGULARIZATION
  //  - WARM_STARTING
  //  - FREEZE_SHAPE_PARAMETERS
  struct Options {
    bool temporalRegularization = false;
    bool warmStarting = false;
    bool freezeShapeParameters = false;
  };

  explicit SMPLOptimizer(SMPLModel *smplModel, CameraModel *cameraModel, const Options &options);

  // Fit SMPL parameters to a single frame
  void fitFrame(const std::vector<Point2D> &keypoints);

  // Step 1: Estimate initial translation and rotation based on torso joints.
  void fitInitialization(const std::vector<Point2D> &keypoints);

  // Step 2: Optimize SMPL parameters
  void fitFull(const std::vector<Point2D> &keypoints);

  // Expose parameters
  const Eigen::Vector3d &getGlobalT() const { return globalT_; }
  const std::vector<double> &getPoseParams() const { return poseParams_; }
  const std::vector<double> &getShapeParams() const { return shapeParams_; }

  // Expose optimization summary
  const ceres::Solver::Summary &getInitSummary() const { return initSummary_; }
  const ceres::Solver::Summary &getFullSummary() const { return fullSummary_; }

private:
  // Configuration flags (see Options above).
  Options options_;

  // Global translation parameters
  Eigen::Vector3d globalT_ = Eigen::Vector3d::Zero();

  // SMPL pose parameters (e.g., 72-dim axis-angle)
  std::vector<double> poseParams_;

  // SMPL shape parameters (e.g., 10 betas)
  std::vector<double> shapeParams_;

  // 2D joints for the current frame
  std::vector<Point2D> current2DJoints;

  // Pointer to SMPL model
  SMPLModel *smplModel_ = nullptr;

  // Pointer to camera model
  CameraModel *cameraModel_ = nullptr;

  // Previous frame
  bool hasPreviousFrame_ = false;
  std::vector<double> prevGlobalT_;
  std::vector<double> prevPoseParams_;
  std::vector<double> prevShapeParams_;

  // Summary for ceres optimizer
  ceres::Solver::Summary initSummary_;
  ceres::Solver::Summary fullSummary_;
};