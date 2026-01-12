// FittingOptimizer.cpp
// Implements the SMPL fitting optimization pipeline.
// Uses Ceres Solver to minimize reprojection error between:
//   - projected 3D SMPL joints
//   - 2D OpenPose keypoints
//
// Contains:
//   - construction of Ceres problem
//   - cost functions for joint reprojection
//   - optimization loop per frame
//
// TODOs:
//   - define cost residuals
//   - implement solveFrame() for per-frame fitting
//   - add initialization heuristics for pose/shape

#include "FittingOptimizer.h"
#include "SMPLModel.h"  // Declared in the header, the implementation will come later
#include "CameraModel.h"
#include <unordered_map>
#include <limits>
#include <cmath>

// SMPL (24) → OpenPose BODY_25 mapping
static const std::unordered_map<int, int> SMPL_TO_OPENPOSE = {
    {0,  8},   // Pelvis -> MidHip
    {1,  9},   // L Hip
    {2, 12},   // R Hip
    {3, 10},   // L Knee
    {4, 13},   // R Knee
    {5, 11},   // L Ankle
    {6, 14},   // R Ankle
    {7,  1},   // Spine -> Neck
    {8,  2},   // L Shoulder
    {9,  5},   // R Shoulder
    {10, 3},   // L Elbow
    {11, 6},   // R Elbow
    {12, 4},   // L Wrist
    {13, 7},   // R Wrist
    {14, 0},   // Head -> Nose
};


FittingOptimizer::FittingOptimizer(SMPLModel* smplModel_,
                                    const Options& options_)
    : options(options_), smplModel(smplModel_)
{
    // TODO: Confirm the dimensionality of SMPL parameters for this project
    // Typical SMPL uses:
    //   - 72 pose parameters (24 joints * 3 axis-angle)
    //   - 10 shape parameters (betas)
    //
    // For now, initialize with reasonable defaults (all zeros)
    int numPose  = 72;
    int numShape = 10;

    if (smplModel && smplModel->isLoaded()) {
        numPose  = smplModel->numPoseCoeffs();
        numShape = smplModel->numShapeCoeffs();
    }

    poseParams.assign(numPose,  0.0);
    shapeParams.assign(numShape, 0.0);

    poseHistory.clear();
    shapeHistory.clear();
}

void FittingOptimizer::fitFrame(const Pose2D& observation)
{
    // Store the 2D joints for this frame
    current2DJoints = observation;

    // TODO:
    //  1) If options.warmStarting is true, initialize poseParams / shapeParams
    //     from the previous frame instead of a fixed A-pose
    //  2) If options.freezeShapeParameters is true, keep shapeParams fixed
    //     across frames and do not update them in the optimizer
    //  3) Build a Ceres problem for this frame (and neighbors if temporal reg. is used)
    //  4) Run the solver and update poseParams/shapeParams with the result

    // Placeholder calls to show the intended flow (no-op for now)
    buildProblemForCurrentFrame();
    addReprojectionTerms();
    addPriorTerms();

    // Store *raw* parameters for this frame into the history buffer
    poseHistory.push_back(poseParams);
    shapeHistory.push_back(shapeParams);

    // Apply temporal regularization (currently implemented as explicit smoothing)
    if (options.temporalRegularization) {
        addTemporalRegularizationTerms();
    }

    // Make sure downstream code (e.g., main.cpp) sees the
    // possibly-smoothed parameters when calling SMPLModel
    if (smplModel) {
        smplModel->setPose(poseParams);
        smplModel->setShape(shapeParams);
    }

    // NOTE: Ceres solve will be added later; temporal smoothing here
    //       conceptually implements a penalty on frame-to-frame changes
}

void FittingOptimizer::buildProblemForCurrentFrame()
{
    // TODO:
    //  - Create a Ceres::Problem instance
    //  - Add parameter blocks for poseParams and shapeParams
    //  - Prepare any fixed data needed for this frame (e.g., 3D joint regressor)
}

void FittingOptimizer::addReprojectionTerms()
{
    // TODO:
    //  - For each 2D joint with sufficient confidence:
    //      * Project the corresponding 3D SMPL joint
    //      * Add a reprojection error residual block
    //  - Define a cost functor that compares projected vs. observed 2D joint

    if (!smplModel || !smplModel->isLoaded()) {
        std::cout << "[Reprojection] SMPL model not loaded\n";
        return;
    }

    // Get 3D SMPL joints (Nx3)
    Eigen::MatrixXf smplJoints = smplModel->getJointPositions();

    // Simple pinhole camera (temporary values)
    CameraModel camera(
        1000.0f, 1000.0f,   // fx, fy
        640.0f / 2.0f,      // cx
        480.0f / 2.0f       // cy
    );

    double totalError = 0.0;
    int validJoints = 0;

    for (const auto& [smplIdx, opIdx] : SMPL_TO_OPENPOSE) {

        if (smplIdx >= smplJoints.rows())
            continue;

        if (opIdx >= static_cast<int>(current2DJoints.keypoints.size()))
            continue;

        const Point2D& kp = current2DJoints.keypoints[opIdx];

        // Skip low-confidence OpenPose joints
        if (kp.score < 0.2f)
            continue;

        Eigen::Vector3f joint3D = smplJoints.row(smplIdx);

        // Skip points behind the camera
        if (joint3D.z() <= 0)
            continue;

        Eigen::Vector2f projected = camera.project(joint3D);

        double dx = projected.x() - kp.x;
        double dy = projected.y() - kp.y;

        double error = std::sqrt(dx * dx + dy * dy);

        // Weight by confidence
        totalError += error * kp.score;
        validJoints++;
    }

    if (validJoints > 0) {
        double avgError = totalError / validJoints;
        std::cout << "[Reprojection] Avg error: "
                  << avgError << " px ("
                  << validJoints << " joints)"
                  << std::endl;
    } else {
        std::cout << "[Reprojection] No valid joints\n";
    }

}

void FittingOptimizer::addPriorTerms()
{
    // TODO:
    //  - Add regularization terms on pose (e.g., pose prior) and shape (beta prior)
    //  - Optionally add joint limit penalties or smoothness constraints
}

void FittingOptimizer::addTemporalRegularizationTerms()
{
    // With a single frame there is nothing to regularize against
    if (poseHistory.size() < 2) {
        return;
    }

    // Smooth pose and shape sequences over time using TemporalSmoother.
    // This is equivalent to adding a temporal smoothness residual that
    // penalizes large frame-to-frame changes
    const auto smoothedPose  = smoother.smoothPoseSequence(poseHistory);
    const auto smoothedShape = smoother.smoothShapeSequence(shapeHistory);

    if (!smoothedPose.empty()) {
        // Use smoothed parameters for the current (latest) frame
        poseParams = smoothedPose.back();
    }

    if (!smoothedShape.empty()) {
        shapeParams = smoothedShape.back();
    }
}
