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
  struct Options {
    bool temporalRegularization = false;
    bool warmStarting = false;
    bool freezeShapeParameters = false;
  };

  // Check if warm starting can be used
  bool useWarmStarting() { 
    return hasPreviousFrame_ && options_.warmStarting; 
  }

  // Check if temporal regularization can be used
  bool useTemporalRegularization() { 
    return hasPreviousFrame_ && options_.temporalRegularization;
  }

  // Check if shape parameters can be freezed
  bool freezeShapeParameters() { 
    return prevShapeParams_.size() > 0 && options_.freezeShapeParameters;
  }

  explicit SMPLOptimizer(SMPLModel *smplModel, CameraModel *cameraModel, const Options &options);

  // Fit SMPL parameters to a single frame
  void fitFrame(const std::vector<Point2D> &keypoints);

  // Step 1: Estimate initial translation and rotation based on torso joints.
  void fitInitialization(const std::vector<Point2D> &keypoints);

  // Step 2: Optimize SMPL parameters
  void fitFull(const std::vector<Point2D> &keypoints);

  // Expose parameters
  const Eigen::Vector3d &getGlobalT() const { return globalT_; }
  const Eigen::VectorXd &getPoseParams() const { return poseParams_; }
  const Eigen::VectorXd &getShapeParams() const { return shapeParams_; }

  // Expose optimization summary
  const ceres::Solver::Summary &getInitSummary() const { return initSummary_; }
  const ceres::Solver::Summary &getFullSummary() const { return fullSummary_; }

private:
  // Configuration flags (see Options above).
  Options options_;

  // Global translation parameters
  Eigen::Vector3d globalT_ = Eigen::Vector3d::Zero();

  // SMPL pose parameters (e.g., 72-dim axis-angle)
  Eigen::VectorXd poseParams_;

  // SMPL shape parameters (e.g., 10 betas)
  Eigen::VectorXd shapeParams_;

  // Pointer to SMPL model
  SMPLModel *smplModel_ = nullptr;

  // Pointer to camera model
  CameraModel *cameraModel_ = nullptr;

  // Previous (t-1) frame
  bool hasPreviousFrame_ = false;
  Eigen::VectorXd prevGlobalT_;
  Eigen::VectorXd prevPoseParams_;
  Eigen::VectorXd prevShapeParams_;
  Eigen::Matrix<double, 24, 3> prevJoints_;

  // Summary for ceres optimizer
  ceres::Solver::Summary initSummary_;
  ceres::Solver::Summary fullSummary_;
};