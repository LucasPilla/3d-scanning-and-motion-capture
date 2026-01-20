// FittingOptimizer
// -----------------
// Performs non-linear optimization (SMPLify-style) to fit SMPL parameters
// to 2D OpenPose detections using Ceres Solver.

#pragma once

#include "CameraModel.h"
#include "PoseDetector.h"
#include "TemporalSmoother.h"
#include <Eigen/Dense>
#include <vector>

// TODO: Implement SMPLModel in SMPLModel.h / SMPLModel.cpp
class SMPLModel;

// Interface for SMPL fitting (no optimization)
//
// This class will later own the Ceres problem setup and perform SMPLify-style
// optimization per frame
class FittingOptimizer
{
public:
    // Configuration flags controlling advanced features
    // These correspond to the proposal:
    //  - TEMPORAL_REGULARIZATION
    //  - WARM_STARTING
    //  - FREEZE_SHAPE_PARAMETERS
    struct Options
    {
        bool temporalRegularization = false;
        bool warmStarting = false;
        bool freezeShapeParameters = false;
    };

    explicit FittingOptimizer(SMPLModel *smplModel, CameraModel *cameraModel,
                              const Options &options);

    // Fit SMPL parameters to a single frame
    std::vector<Point2D> fitFrame(const Pose2D &observation);

    // Step 1: Fit a global 3D rigid transform (R, t) in camera space so that
    // SMPL 3D joints align better with OpenPose 2D detections
    void fitRigid(const Pose2D &observation);

    // Step 2: Optimize SMPL pose parameters using reprojection error
    void fitPose(const Pose2D &observation, std::vector<Point2D> &smpl2DOut);

private:
    // Global rigid transform computed in Step 1 (fitRigid)
    Eigen::Matrix3d globalR_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d globalT_ = Eigen::Vector3d::Zero();

    // Configuration flags (see Options above).
    Options options;

    // ---------- Stored data ----------

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
};