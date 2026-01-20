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
#include "SMPLModel.h"
#include "CameraModel.h"
#include <unordered_map>
#include <limits>
#include <cmath>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct RigidReprojectionResidual
{
    RigidReprojectionResidual(float X, float Y, float Z,
                              float x_obs, float y_obs,
                              float fx, float fy, float cx, float cy,
                              float weight)
        : X_(X), Y_(Y), Z_(Z),
          x_obs_(x_obs), y_obs_(y_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          w_(weight)
    {}

    template <typename T>
    bool operator()(const T* const pose, T* residuals) const
    {
        // pose[0,1,2] = angle-axis vector (magnitude is angle, direction is axis)
        // pose[3,4,5] = translation
        
        const T p_model[3] = {T(X_), T(Y_), T(Z_)};
        T p_cam[3];

        // Ceres helper for Axis-Angle rotation
        ceres::AngleAxisRotatePoint(pose, p_model, p_cam);

        // Add translation
        p_cam[0] += pose[3];
        p_cam[1] += pose[4];
        p_cam[2] += pose[5];

        const T z_inv = T(1.0) / (p_cam[2] + T(1e-8));

        // Pinhole projection
        const T u = T(fx_) * (p_cam[0] * z_inv) + T(cx_);
        const T v = T(fy_) * (p_cam[1] * z_inv) + T(cy_);

        // Residuals
        residuals[0] = T(w_) * (u - T(x_obs_));
        residuals[1] = T(w_) * (v - T(y_obs_));
        return true;
    }

    float X_, Y_, Z_;
    float x_obs_, y_obs_;
    float fx_, fy_, cx_, cy_;
    float w_;
};

// SMPL (24) → OpenPose BODY_25
// Indices based on standard SMPL and OpenPose BODY_25 conventions
static const std::unordered_map<int, int> SMPL_TO_OPENPOSE = {
    // Core
    {0,  8},   // Pelvis        → MidHip
    {12, 1},   // Neck          → Neck

    // Left leg
    {1,  12},  // Left hip      → LHip
    {4,  13},  // Left knee     → LKnee
    {7,  14},  // Left ankle    → LAnkle

    // Right leg
    {2,  9},   // Right hip     → RHip
    {5,  10},  // Right knee    → RKnee
    {8,  11},  // Right ankle   → RAnkle

    // Left arm
    {16, 5},   // Left shoulder → LShoulder
    {18, 6},   // Left elbow    → LElbow
    {20, 7},   // Left wrist    → LWrist

    // Right arm
    {17, 2},   // Right shoulder → RShoulder
    {19, 3},   // Right elbow    → RElbow
    {21, 4}    // Right wrist    → RWrist
};



FittingOptimizer::FittingOptimizer(
    SMPLModel* smplModel_,
    CameraModel* cameraModel_,
    const Options& options_)
    : smplModel(smplModel_), cameraModel(cameraModel_), options(options_) 
{
    // TODO: Confirm the dimensionality of SMPL parameters for this project
    // Typical SMPL uses:
    //   - 72 pose parameters (24 joints * 3 axis-angle)
    //   - 10 shape parameters (betas)
    //
    // For now, initialize with reasonable defaults (all zeros)
    int numPose  = 72;
    int numShape = 10;

    poseParams.assign(numPose,  0.0);
    shapeParams.assign(numShape, 0.0);

    poseHistory.clear();
    shapeHistory.clear();
}

std::vector<Point2D> FittingOptimizer::fitFrame(const Pose2D& observation)
{
    // The following optimization pipeline is based on SMPLify paper
    
    // Step 1 
    std::vector<Point2D> smplProjectedOptimized;
    this->fitRigid3D(observation, smplProjectedOptimized);

    // Step 2

    // Step 3

    return smplProjectedOptimized;
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



// Check Section 3.3 "Optimization" of BogoECCV2016.pdf
// "We estimate the depth via the ratio of similar triangles, defined by the torso length..." 
// "we further refine this estimate by minimizing E_J over the torso joints alone" 

const std::unordered_set<int> TORSO_SMPL_IDS = {1, 2, 16, 17}; 

void FittingOptimizer::fitRigid3D (
    const Pose2D& observation,
    std::vector<Point2D>& smpl2DOut
)
{
    // Retrieve current 3D joints
    Eigen::MatrixXd smplJoints = smplModel->getJointPositions();

    smpl2DOut.assign(smplJoints.rows(), Point2D{});

    // Initialization via Similar Triangles
    // We try to align the torso in 3D to the torso in 2D to guess depth (Tz)

    std::vector<double> torso3D_y;
    std::vector<double> torso2D_y;
    double sum3D_x = 0, sum3D_y = 0;
    double sum2D_x = 0, sum2D_y = 0;
    int count = 0;

    for (const auto& [smplIdx, opIdx] : SMPL_TO_OPENPOSE) {

        if (TORSO_SMPL_IDS.find(smplIdx) == TORSO_SMPL_IDS.end()) continue;

        if (smplIdx >= smplJoints.rows()) continue;
        if (opIdx >= static_cast<int>(observation.keypoints.size())) continue;

        const Point2D& kp = observation.keypoints[opIdx];

        // Require high confidence for initialization
        if (kp.score < 0.4f) continue; 

        torso3D_y.push_back(smplJoints(smplIdx, 1));
        torso2D_y.push_back(kp.y);

        sum3D_x += smplJoints(smplIdx, 0);
        sum3D_y += smplJoints(smplIdx, 1);
        
        sum2D_x += kp.x;
        sum2D_y += kp.y;
        count++;
    }

    // Default guess
    double init_tx = 0.0, init_ty = 0.0, init_tz = 2.50;

    if (count >= 2) {
        // Estimate Depth (Z) based on torso height ratio
        auto mm3D = std::minmax_element(torso3D_y.begin(), torso3D_y.end());
        double h3D = *mm3D.second - *mm3D.first;

        auto mm2D = std::minmax_element(torso2D_y.begin(), torso2D_y.end());
        double h2D = *mm2D.second - *mm2D.first;

        if (h2D > 1.0) {
            init_tz = cameraModel->intrinsics().fy * (h3D / h2D);
        }

        // Estimate Translation (X, Y) by aligning centroids
        double c3D_x = sum3D_x / count;
        double c3D_y = sum3D_y / count;
        double c2D_x = sum2D_x / count;
        double c2D_y = sum2D_y / count;

        // X = (u - cx) * Z / fx
        double center_cam_x = (c2D_x - cameraModel->intrinsics().cx) * init_tz / cameraModel->intrinsics().fx;
        double center_cam_y = (c2D_y - cameraModel->intrinsics().cy) * init_tz / cameraModel->intrinsics().fy;

        init_tx = center_cam_x - c3D_x;
        init_ty = center_cam_y - c3D_y;
    }

    // Torso-Only Optimization
    ceres::Problem problem;
    
    // pose = [rx, ry, rz, tx, ty, tz] (Axis-Angle + Translation)
    double pose[6] = {3.1415926535, 0.0, 0.0, init_tx, init_ty, init_tz};

    for (const auto& [smplIdx, opIdx] : SMPL_TO_OPENPOSE) {

        // Strictly limit to Torso for this stage
        if (TORSO_SMPL_IDS.find(smplIdx) == TORSO_SMPL_IDS.end()) continue;

        if (smplIdx >= smplJoints.rows()) continue;
        if (opIdx >= static_cast<int>(observation.keypoints.size())) continue;

        const Point2D& kp = observation.keypoints[opIdx];
        if (kp.score < 0.2f) continue;

        // Use Huber Loss to handle outliers
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<RigidReprojectionResidual, 2, 6>(
                new RigidReprojectionResidual(
                    smplJoints(smplIdx, 0), 
                    smplJoints(smplIdx, 1), 
                    smplJoints(smplIdx, 2),
                    kp.x, kp.y,
                    cameraModel->intrinsics().fx, 
                    cameraModel->intrinsics().fy, 
                    cameraModel->intrinsics().cx, 
                    cameraModel->intrinsics().cy,
                    std::sqrt(kp.score) // Weight
                )
            );

        problem.AddResidualBlock(cost, loss, pose);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false; 
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // DEBUG: Reproject and Store Results 

    // Apply the optimized transform for visualization
    Eigen::Vector3d r_vec(pose[0], pose[1], pose[2]);
    Eigen::Vector3d t_vec(pose[3], pose[4], pose[5]);
    
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (r_vec.squaredNorm() > 1e-8) {
        R = Eigen::AngleAxisd(r_vec.norm(), r_vec.normalized()).toRotationMatrix();
    }

    for (int i = 0; i < smplJoints.rows(); ++i) {
        Eigen::Vector3d pModel = smplJoints.row(i);
        Eigen::Vector3d pCam = R * pModel + t_vec;

        double Z = pCam.z();
        if (Z < 0.1) Z = 0.1; // Clamp near plane

        Point2D pt;
        pt.x = static_cast<float>(cameraModel->intrinsics().fx * (pCam.x() / Z) + cameraModel->intrinsics().cx);
        pt.y = static_cast<float>(cameraModel->intrinsics().fy * (pCam.y() / Z) + cameraModel->intrinsics().cy);
        pt.score = 1.0f; 
        smpl2DOut[i] = pt;
    }
}