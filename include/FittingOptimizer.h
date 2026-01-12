// FittingOptimizer
// -----------------
// Performs non-linear optimization (SMPLify-style) to fit SMPL parameters
// to 2D OpenPose detections using Ceres Solver.

// Responsibilities:
//  - Define cost functions for joint reprojection
//  - Initialize pose & shape
//  - Run optimization for each frame

// Used by:
//  - Visualization

#pragma once

#include <vector>
#include "PoseDetector.h"
#include "TemporalSmoother.h"

// TODO: Implement SMPLModel in SMPLModel.h / SMPLModel.cpp
class SMPLModel;

// Interface for SMPL fitting (no optimization)
//
// This class will later own the Ceres problem setup and perform SMPLify-style optimization per frame
class FittingOptimizer
{
public:
    // Configuration flags controlling advanced features
    // These correspond to the proposal:
    //  - TEMPORAL_REGULARIZATION
    //  - WARM_STARTING
    //  - FREEZE_SHAPE_PARAMETERS
    struct Options {
        bool temporalRegularization   = false;
        bool warmStarting             = false;
        bool freezeShapeParameters    = false;
    };

    explicit FittingOptimizer(SMPLModel* smplModel,
                              const Options& options);

    // Fit SMPL parameters to a single frame of 2D joints
    //
    // NOTE: In this step, this function will not run any optimization
    // It will just prepare data & placeholders
    void fitFrame(const Pose2D& observation);

    // TODO: Add getters to retrieve the fitted pose/shape
    // for the last processed frame, e.g.:
    //   const std::vector<double>& getPoseParams() const;
    //   const std::vector<double>& getShapeParams() const;

private:
    // Configuration flags (see Options above).
    Options options;

    // ---------- Stored data ----------

    // 2D joints for the current frame
    Pose2D current2DJoints; // TODO: extend to support sequences if needed

    // SMPL pose parameters (e.g., 72-dim axis-angle)
    std::vector<double> poseParams;

    // SMPL shape parameters (e.g., 10 betas)
    std::vector<double> shapeParams;

    // Pointer to SMPL model used for projecting 3D joints
    SMPLModel* smplModel = nullptr;

    // History of parameters for temporal smoothing / regularization
    TemporalSmoother smoother;
    TemporalSmoother::ParamSequence poseHistory;
    TemporalSmoother::ParamSequence shapeHistory;

    // ---------- Ceres preparation hooks (no implementation yet) ----------

    // TODO: Build the Ceres problem for the current frame (create residuals, etc.)
    void buildProblemForCurrentFrame();

    // TODO: Add reprojection error residuals (2D joints vs projected SMPL joints)
    void addReprojectionTerms();

    // TODO: Add pose/shape priors and regularization terms
    void addPriorTerms();

    // TODO: Add temporal regularization terms (smoothing during optimization),
    // potentially using TemporalSmoother utilities.
    void addTemporalRegularizationTerms();
};