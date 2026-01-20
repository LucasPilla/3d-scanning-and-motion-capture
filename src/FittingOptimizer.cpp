// FittingOptimizer.cpp
// Implements the SMPL fitting optimization pipeline.

#include "FittingOptimizer.h"
#include "CameraModel.h"
#include "SMPLModel.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <limits>
#include <unordered_map>

// Cost functor for Step 1
struct RigidReprojectionResidual
{
    RigidReprojectionResidual(float X, float Y, float Z, float x_obs, float y_obs,
                              float fx, float fy, float cx, float cy,
                              float weight)
        : X_(X), Y_(Y), Z_(Z), x_obs_(x_obs), y_obs_(y_obs), fx_(fx), fy_(fy),
          cx_(cx), cy_(cy), w_(weight) {}

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
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

// Cost functor for Step 2 
struct PoseReprojectionCost
{
    PoseReprojectionCost(int smplJointIdx, double x_obs, double y_obs,
                             double weight, double fx, double fy, double cx,
                             double cy, const Eigen::Matrix3d &globalR,
                             const Eigen::Vector3d &globalT,
                             const Eigen::Matrix<double, 24, 3> &J_rest,
                             const Eigen::VectorXi &parents)
        : smplJointIdx_(smplJointIdx), x_obs_(x_obs), y_obs_(y_obs),
          weight_(weight), fx_(fx), fy_(fy), cx_(cx), cy_(cy), globalR_(globalR),
          globalT_(globalT), J_rest_(J_rest), parents_(parents) {}

    template <typename T>
    bool operator()(const T *const poseParamsPtr, T *residuals) const
    {
        // Convert raw pointer to fixed-size Eigen vector (no heap allocation)
        Eigen::Matrix<T, 72, 1> poseParams;
        for (int i = 0; i < 72; ++i)
        {
            poseParams(i) = poseParamsPtr[i];
        }

        // Forward kinematics 
        Eigen::Matrix<T, 24, 3> jointPositions =
            SMPLModel::forwardKinematics<T>(poseParams, J_rest_, parents_);

        // Get this joint's position
        Eigen::Matrix<T, 3, 1> jointPos =
            jointPositions.row(smplJointIdx_).transpose();

        // Apply the global rigid transform (from Step 1)
        Eigen::Matrix<T, 3, 3> R;
        R << T(globalR_(0, 0)), T(globalR_(0, 1)), T(globalR_(0, 2)),
            T(globalR_(1, 0)), T(globalR_(1, 1)), T(globalR_(1, 2)),
            T(globalR_(2, 0)), T(globalR_(2, 1)), T(globalR_(2, 2));

        Eigen::Matrix<T, 3, 1> t;
        t << T(globalT_(0)), T(globalT_(1)), T(globalT_(2));

        Eigen::Matrix<T, 3, 1> pCam = R * jointPos + t;

        // Project to 2D using pinhole camera model
        T z_inv = T(1.0) / (pCam(2) + T(1e-8));
        T u = T(fx_) * pCam(0) * z_inv + T(cx_);
        T v = T(fy_) * pCam(1) * z_inv + T(cy_);

        // Weighted reprojection residuals
        residuals[0] = T(weight_) * (u - T(x_obs_));
        residuals[1] = T(weight_) * (v - T(y_obs_));

        return true;
    }

    int smplJointIdx_;
    double x_obs_, y_obs_, weight_;
    double fx_, fy_, cx_, cy_;
    Eigen::Matrix3d globalR_;
    Eigen::Vector3d globalT_;
    Eigen::Matrix<double, 24, 3> J_rest_; // Pre-computed rest-pose joints
    Eigen::VectorXi parents_;             // Kinematic tree parents
};

// SMPL (24) → OpenPose BODY_25
// Indices based on standard SMPL and OpenPose BODY_25 conventions
static const std::unordered_map<int, int> SMPL_TO_OPENPOSE = {
    // Core
    {0, 8},  // Pelvis        → MidHip
    {12, 1}, // Neck          → Neck

    // Left leg
    {1, 12}, // Left hip      → LHip
    {4, 13}, // Left knee     → LKnee
    {7, 14}, // Left ankle    → LAnkle

    // Right leg
    {2, 9},  // Right hip     → RHip
    {5, 10}, // Right knee    → RKnee
    {8, 11}, // Right ankle   → RAnkle

    // Left arm
    {16, 5}, // Left shoulder → LShoulder
    {18, 6}, // Left elbow    → LElbow
    {20, 7}, // Left wrist    → LWrist

    // Right arm
    {17, 2}, // Right shoulder → RShoulder
    {19, 3}, // Right elbow    → RElbow
    {21, 4}  // Right wrist    → RWrist
};

FittingOptimizer::FittingOptimizer(SMPLModel *smplModel_,
                                   CameraModel *cameraModel_,
                                   const Options &options_)
    : smplModel(smplModel_), cameraModel(cameraModel_), options(options_)
{
    int numPose = 72;
    int numShape = 10;

    poseParams.assign(numPose, 0.0);
    shapeParams.assign(numShape, 0.0);

    poseHistory.clear();
    shapeHistory.clear();
}

std::vector<Point2D> FittingOptimizer::fitFrame(const Pose2D &observation)
{
    // The following optimization pipeline is based on SMPLify paper

    // Step 1: Fit global rigid transform (R, t) using torso joints
    this->fitRigid(observation);

    // Step 2: Optimize SMPL pose parameters using all visible joints
    std::vector<Point2D> smplProjectedOptimized;
    this->fitPose(observation, smplProjectedOptimized);

    // Step 3: TODO

    return smplProjectedOptimized;
}

// Check Section 3.3 "Optimization" of BogoECCV2016.pdf
// "We estimate the depth via the ratio of similar triangles, defined by the
// torso length..." "we further refine this estimate by minimizing E_J over the
// torso joints alone"

const std::unordered_set<int> TORSO_SMPL_IDS = {1, 2, 16, 17};

void FittingOptimizer::fitRigid(const Pose2D &observation)
{
    // Retrieve current 3D joints
    Eigen::MatrixXd smplJoints = smplModel->getJointPositions();

    // Initialization via Similar Triangles
    // We try to align the torso in 3D to the torso in 2D to guess depth (Tz)

    std::vector<double> torso3D_y;
    std::vector<double> torso2D_y;
    double sum3D_x = 0, sum3D_y = 0;
    double sum2D_x = 0, sum2D_y = 0;
    int count = 0;

    for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
    {

        if (TORSO_SMPL_IDS.find(smplIdx) == TORSO_SMPL_IDS.end())
            continue;

        if (smplIdx >= smplJoints.rows())
            continue;
        if (opIdx >= static_cast<int>(observation.keypoints.size()))
            continue;

        const Point2D &kp = observation.keypoints[opIdx];

        // Require high confidence for initialization
        if (kp.score < 0.4f)
            continue;

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

    if (count >= 2)
    {
        // Estimate Depth (Z) based on torso height ratio
        auto mm3D = std::minmax_element(torso3D_y.begin(), torso3D_y.end());
        double h3D = *mm3D.second - *mm3D.first;

        auto mm2D = std::minmax_element(torso2D_y.begin(), torso2D_y.end());
        double h2D = *mm2D.second - *mm2D.first;

        if (h2D > 1.0)
        {
            init_tz = cameraModel->intrinsics().fy * (h3D / h2D);
        }

        // Estimate Translation (X, Y) by aligning centroids
        double c3D_x = sum3D_x / count;
        double c3D_y = sum3D_y / count;
        double c2D_x = sum2D_x / count;
        double c2D_y = sum2D_y / count;

        // X = (u - cx) * Z / fx
        double center_cam_x = (c2D_x - cameraModel->intrinsics().cx) * init_tz /
                              cameraModel->intrinsics().fx;
        double center_cam_y = (c2D_y - cameraModel->intrinsics().cy) * init_tz /
                              cameraModel->intrinsics().fy;

        init_tx = center_cam_x - c3D_x;
        init_ty = center_cam_y - c3D_y;
    }

    // Torso-Only Optimization
    ceres::Problem problem;

    // pose = [rx, ry, rz, tx, ty, tz] (Axis-Angle + Translation)
    double pose[6] = {3.1415926535, 0.0, 0.0, init_tx, init_ty, init_tz};

    for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
    {

        // Strictly limit to Torso for this stage
        if (TORSO_SMPL_IDS.find(smplIdx) == TORSO_SMPL_IDS.end())
            continue;

        if (smplIdx >= smplJoints.rows())
            continue;
        if (opIdx >= static_cast<int>(observation.keypoints.size()))
            continue;

        const Point2D &kp = observation.keypoints[opIdx];
        if (kp.score < 0.2f)
            continue;

        // Use Huber Loss to handle outliers
        ceres::LossFunction *loss = new ceres::HuberLoss(1.0);

        ceres::CostFunction *cost =
            new ceres::AutoDiffCostFunction<RigidReprojectionResidual, 2, 6>(
                new RigidReprojectionResidual(
                    smplJoints(smplIdx, 0), smplJoints(smplIdx, 1),
                    smplJoints(smplIdx, 2), kp.x, kp.y,
                    cameraModel->intrinsics().fx, cameraModel->intrinsics().fy,
                    cameraModel->intrinsics().cx, cameraModel->intrinsics().cy,
                    std::sqrt(kp.score) // Weight
                    ));

        problem.AddResidualBlock(cost, loss, pose);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    // Apply the optimized transform for visualization
    Eigen::Vector3d r_vec(pose[0], pose[1], pose[2]);
    Eigen::Vector3d t_vec(pose[3], pose[4], pose[5]);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (r_vec.squaredNorm() > 1e-8)
    {
        R = Eigen::AngleAxisd(r_vec.norm(), r_vec.normalized()).toRotationMatrix();
    }

    // Store for Step 2
    globalR_ = R;
    globalT_ = t_vec;
}

void FittingOptimizer::fitPose(const Pose2D &observation,
                               std::vector<Point2D> &smpl2DOut)
{
    // Get SMPL model data for the cost functor
    const Eigen::MatrixXd &templateVertices = smplModel->getTemplateVertices();
    const Eigen::MatrixXd &shapeBlendShapes = smplModel->getShapeBlendShapes();
    const Eigen::MatrixXd &jointRegressor = smplModel->getJointRegressor();
    const Eigen::MatrixXi &kinematicTree = smplModel->getKinematicTree();

    // Current shape parameters (fixed for now)
    Eigen::VectorXd shapeParams = Eigen::Map<Eigen::VectorXd>(
        this->shapeParams.data(), static_cast<int>(this->shapeParams.size()));

    // ====================================================================
    // OPTIMIZATION: Pre-compute rest-pose joints (done once, not in loop)
    // ====================================================================
    int N = static_cast<int>(templateVertices.rows());
    int numShapeCoeffs = static_cast<int>(shapeParams.size());

    // Compute shaped vertices
    Eigen::MatrixXd v_shaped = templateVertices;
    if (numShapeCoeffs > 0 && shapeBlendShapes.cols() == numShapeCoeffs)
    {
        Eigen::VectorXd shape_offset = shapeBlendShapes * shapeParams;
        for (int i = 0; i < N; ++i)
        {
            v_shaped(i, 0) += shape_offset(i * 3 + 0);
            v_shaped(i, 1) += shape_offset(i * 3 + 1);
            v_shaped(i, 2) += shape_offset(i * 3 + 2);
        }
    }

    // Regress rest-pose joints (fixed-size for optimization)
    Eigen::MatrixXd J_dynamic = jointRegressor * v_shaped;
    Eigen::Matrix<double, 24, 3> J_rest;
    for (int j = 0; j < 24; ++j)
    {
        J_rest.row(j) = J_dynamic.row(j);
    }

    // Get kinematic tree parents
    Eigen::VectorXi parents = kinematicTree.row(0);

    // Initialize pose parameters (72 values = 24 joints * 3 axis-angle)
    // Start from current poseParams or zero
    std::vector<double> poseOptim(72, 0.0);
    for (size_t i = 0; i < std::min(poseParams.size(), poseOptim.size()); ++i)
    {
        poseOptim[i] = poseParams[i];
    }

    ceres::Problem problem;

    // Add reprojection residuals for all visible joints
    for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
    {
        if (opIdx >= static_cast<int>(observation.keypoints.size()))
            continue;

        const Point2D &kp = observation.keypoints[opIdx];
        if (kp.score < 0.2f)
            continue;

        ceres::LossFunction *loss = new ceres::HuberLoss(1.0);

        ceres::CostFunction *cost =
            new ceres::AutoDiffCostFunction<PoseReprojectionCost, 2, 72>(
                new PoseReprojectionCost(
                    smplIdx, kp.x, kp.y, std::sqrt(kp.score),
                    cameraModel->intrinsics().fx, cameraModel->intrinsics().fy,
                    cameraModel->intrinsics().cx, cameraModel->intrinsics().cy,
                    globalR_, globalT_, J_rest, parents));

        problem.AddResidualBlock(cost, loss, poseOptim.data());
    }

    // Solver options
    ceres::Solver::Options solverOptions;
    solverOptions.max_num_iterations = 100;
    solverOptions.linear_solver_type = ceres::DENSE_QR;
    solverOptions.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions, &problem, &summary);

    // Store optimized pose parameters
    poseParams = poseOptim;

    // Update SMPL model with new pose
    smplModel->setPose(poseParams);

    // Reproject joints for visualization using optimized forward kinematics
    constexpr int numJoints = 24;

    // Build rotation matrices
    std::vector<Eigen::Matrix3d> rotations(numJoints);
    for (int j = 0; j < numJoints; ++j)
    {
        Eigen::Vector3d r(poseOptim[j * 3], poseOptim[j * 3 + 1],
                          poseOptim[j * 3 + 2]);
        rotations[j] = SMPLModel::rodrigues<double>(r);
    }

    // Forward kinematics (reuse J_rest and parents from earlier)
    std::vector<Eigen::Matrix3d> globalRot(numJoints);
    std::vector<Eigen::Vector3d> globalTrans(numJoints);

    globalRot[0] = rotations[0];
    globalTrans[0] = J_rest.row(0).transpose();

    for (int j = 1; j < numJoints; ++j)
    {
        int p = parents(j);
        Eigen::Vector3d localT = (J_rest.row(j) - J_rest.row(p)).transpose();
        globalRot[j] = globalRot[p] * rotations[j];
        globalTrans[j] = globalTrans[p] + globalRot[p] * localT;
    }

    // Project each joint to 2D
    smpl2DOut.resize(numJoints);
    for (int j = 0; j < numJoints; ++j)
    {
        Eigen::Vector3d pCam = globalR_ * globalTrans[j] + globalT_;

        double Z = pCam.z();
        if (Z < 0.1)
            Z = 0.1;

        Point2D pt;
        pt.x = static_cast<float>(cameraModel->intrinsics().fx * (pCam.x() / Z) +
                                  cameraModel->intrinsics().cx);
        pt.y = static_cast<float>(cameraModel->intrinsics().fy * (pCam.y() / Z) +
                                  cameraModel->intrinsics().cy);
        pt.score = 1.0f;
        smpl2DOut[j] = pt;
    }
}