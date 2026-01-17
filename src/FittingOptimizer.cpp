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
#include <ceres/ceres.h>

struct TranslationResidual
{
    TranslationResidual(float x_smpl,
                        float y_smpl,
                        float x_obs,
                        float y_obs,
                        float weight)
        : x_smpl_(x_smpl),
          y_smpl_(y_smpl),
          x_obs_(x_obs),
          y_obs_(y_obs),
          w_(weight)
    {}

    template <typename T>
    bool operator()(const T* const t, T* residuals) const
    {
        // t[0] = dx, t[1] = dy
        T u = T(x_smpl_) + t[0];
        T v = T(y_smpl_) + t[1];

        residuals[0] = T(w_) * (u - T(x_obs_));
        residuals[1] = T(w_) * (v - T(y_obs_));
        return true;
    }

    float x_smpl_, y_smpl_;
    float x_obs_, y_obs_;
    float w_;
};

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
        // pose = [rx, ry, rz, tx, ty, tz] (Euler angles + translation)
        const T rx = pose[0];
        const T ry = pose[1];
        const T rz = pose[2];
        const T tx = pose[3];
        const T ty = pose[4];
        const T tz = pose[5];

        const T cxr = ceres::cos(rx);
        const T sxr = ceres::sin(rx);
        const T cyr = ceres::cos(ry);
        const T syr = ceres::sin(ry);
        const T czr = ceres::cos(rz);
        const T szr = ceres::sin(rz);

        // R = Rz * Ry * Rx
        const T R00 = czr * cyr;
        const T R01 = czr * syr * sxr - szr * cxr;
        const T R02 = czr * syr * cxr + szr * sxr;
        const T R10 = szr * cyr;
        const T R11 = szr * syr * sxr + czr * cxr;
        const T R12 = szr * syr * cxr - czr * sxr;
        const T R20 = -syr;
        const T R21 = cyr * sxr;
        const T R22 = cyr * cxr;

        const T X = T(X_);
        const T Y = T(Y_);
        const T Z = T(Z_);

        const T Xr = R00 * X + R01 * Y + R02 * Z;
        const T Yr = R10 * X + R11 * Y + R12 * Z;
        const T Zr = R20 * X + R21 * Y + R22 * Z;

        const T Xc = Xr + tx;
        const T Yc = Yr + ty;
        const T Zc = Zr + tz;

        const T Zsafe = Zc + T(1e-6); // avoid divide-by-zero

        const T u = T(fx_) * (Xc / Zsafe) + T(cx_);
        const T v = T(fy_) * (Yc / Zsafe) + T(cy_);

        const T w = T(w_);
        residuals[0] = w * (u - T(x_obs_));
        residuals[1] = w * (v - T(y_obs_));
        return true;
    }

    float X_, Y_, Z_;
    float x_obs_, y_obs_;
    float fx_, fy_, cx_, cy_;
    float w_;
};

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

void FittingOptimizer::fit2DTranslation(const std::vector<Point2D>& smpl2D,
                                        double& outDx,
                                        double& outDy)
{
    outDx = 0.0;
    outDy = 0.0;

    if (current2DJoints.keypoints.empty()) {
        std::cout << "[TranslationFit] No OpenPose joints for this frame.\n";
        return;
    }

    ceres::Problem problem;

    double t[2] = {0.0, 0.0}; // dx, dy

    for (const auto& [smplIdx, opIdx] : SMPL_TO_OPENPOSE) {
        if (smplIdx < 0 || smplIdx >= static_cast<int>(smpl2D.size())) {
            continue;
        }
        if (opIdx < 0 || opIdx >= static_cast<int>(current2DJoints.keypoints.size())) {
            continue;
        }

        const Point2D& smplPt = smpl2D[smplIdx];
        const Point2D& obsPt  = current2DJoints.keypoints[opIdx];

        // Skip if either side is invalid or low confidence
        if (smplPt.x <= 0.0f || smplPt.y <= 0.0f) {
            continue;
        }
        if (obsPt.score < 0.2f || obsPt.x <= 0.0f || obsPt.y <= 0.0f) {
            continue;
        }

        double weight = static_cast<double>(std::sqrt(obsPt.score));

        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<TranslationResidual, 2, 2>(
                new TranslationResidual(
                    smplPt.x, smplPt.y,
                    obsPt.x,  obsPt.y,
                    static_cast<float>(weight))
            );

        problem.AddResidualBlock(cost, nullptr, t);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true; // see loss per iteration

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    outDx = t[0];
    outDy = t[1];
}

void FittingOptimizer::fitRigid3D(const Eigen::MatrixXf& smplJointsCam,
                                  float fx, float fy, float cx, float cy,
                                  std::vector<Point2D>& smpl2DOut)
{
    smpl2DOut.assign(smplJointsCam.rows(), Point2D{});

    if (current2DJoints.keypoints.empty()) {
        std::cout << "[RigidFit] No OpenPose joints for this frame.\n";
        return;
    }

    ceres::Problem problem;

    // pose = [rx, ry, rz, tx, ty, tz]
    double pose[6] = {0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0};

    for (const auto& [smplIdx, opIdx] : SMPL_TO_OPENPOSE) {
        if (smplIdx < 0 || smplIdx >= smplJointsCam.rows()) {
            continue;
        }
        if (opIdx < 0 || opIdx >= static_cast<int>(current2DJoints.keypoints.size())) {
            continue;
        }

        const Point2D& kp = current2DJoints.keypoints[opIdx];
        if (kp.score < 0.2f) {
            continue;
        }

        const float X = smplJointsCam(smplIdx, 0);
        const float Y = smplJointsCam(smplIdx, 1);
        const float Z = smplJointsCam(smplIdx, 2);

        const double weight = std::sqrt(static_cast<double>(kp.score));

        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<RigidReprojectionResidual, 2, 6>(
                new RigidReprojectionResidual(
                    X, Y, Z,
                    kp.x, kp.y,
                    fx, fy, cx, cy,
                    static_cast<float>(weight))
            );

        problem.AddResidualBlock(cost, nullptr, pose);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    // Reproject all SMPL joints with the optimized pose
    const double rx = pose[0], ry = pose[1], rz = pose[2];
    const double tx = pose[3], ty = pose[4], tz = pose[5];

    const double cxr = std::cos(rx);
    const double sxr = std::sin(rx);
    const double cyr = std::cos(ry);
    const double syr = std::sin(ry);
    const double czr = std::cos(rz);
    const double szr = std::sin(rz);

    const double R00 = czr * cyr;
    const double R01 = czr * syr * sxr - szr * cxr;
    const double R02 = czr * syr * cxr + szr * sxr;
    const double R10 = szr * cyr;
    const double R11 = szr * syr * sxr + czr * cxr;
    const double R12 = szr * syr * cxr - czr * sxr;
    const double R20 = -syr;
    const double R21 = cyr * sxr;
    const double R22 = cyr * cxr;

    for (int i = 0; i < smplJointsCam.rows(); ++i) {
        const double X = smplJointsCam(i, 0);
        const double Y = smplJointsCam(i, 1);
        const double Z = smplJointsCam(i, 2);

        const double Xr = R00 * X + R01 * Y + R02 * Z;
        const double Yr = R10 * X + R11 * Y + R12 * Z;
        const double Zr = R20 * X + R21 * Y + R22 * Z;

        const double Xc = Xr + tx;
        const double Yc = Yr + ty;
        const double Zc = Zr + tz;

        const double Zsafe = Zc + 1e-6;

        const double u = fx * (Xc / Zsafe) + cx;
        const double v = fy * (Yc / Zsafe) + cy;

        Point2D pt;
        pt.x = static_cast<float>(u);
        pt.y = static_cast<float>(v);
        pt.score = 1.0f;
        smpl2DOut[i] = pt;
    }
}