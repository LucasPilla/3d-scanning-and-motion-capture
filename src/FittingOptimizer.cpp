// FittingOptimizer.cpp
// Implements the SMPL fitting optimization pipeline.

#include "FittingOptimizer.h"
#include "CameraModel.h"
#include "SMPLModel.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_map>

// SMPL (24) -> OpenPose BODY_25 Mapping
static const std::unordered_map<int, int> SMPL_TO_OPENPOSE = {
	{0, 8}, {12, 1}, // Torso
	{1, 12},
	{4, 13},
	{7, 14}, // Left leg
	{2, 9},
	{5, 10},
	{8, 11}, // Right leg
	{16, 5},
	{18, 6},
	{20, 7}, // Left arm
	{17, 2},
	{19, 3},
	{21, 4} // Right arm
};

static const std::unordered_set<int> TORSO_SMPL_IDS = {1, 2, 16, 17};

// Residual for step 1
struct RigidReprojectionResidual
{
	RigidReprojectionResidual(float X, float Y, float Z, const Point2D &keypoint,
							  const CameraModel &camera)
		: X_(X), Y_(Y), Z_(Z), keypoint_(keypoint), camera_(camera) {}

	template <typename T>
	bool operator()(const T *const pose, T *residuals) const
	{
		// pose[0-2]: Angle-Axis rotation
		// pose[3-5]: Translation

		T p_in[3] = {T(X_), T(Y_), T(Z_)};
		T p_cam[3];

		ceres::AngleAxisRotatePoint(pose, p_in, p_cam);

		p_cam[0] += pose[3];
		p_cam[1] += pose[4];
		p_cam[2] += pose[5];

		// Pinhole projection
		const auto &K = camera_.intrinsics();
		const T z_inv = T(1.0) / (p_cam[2] + T(1e-8));

		const T u = T(K.fx) * (p_cam[0] * z_inv) + T(K.cx);
		const T v = T(K.fy) * (p_cam[1] * z_inv) + T(K.cy);

		// Weighted Residuals
		const T weight = T(std::sqrt(keypoint_.score));
		residuals[0] = weight * (u - T(keypoint_.x));
		residuals[1] = weight * (v - T(keypoint_.y));

		return true;
	}

	// 3D point in the SMPL model
	float X_, Y_, Z_;

	// 2D point in image
	const Point2D &keypoint_;

	// Camera model with intrinsic parameters for 3D->2D projection.
	const CameraModel &camera_;
};

// Residual for step 2
struct PoseReprojectionCost
{
	PoseReprojectionCost(int smplJointIdx, const Point2D &keypoint,
						 const CameraModel &camera,
						 const Eigen::Matrix3d &globalR,
						 const Eigen::Vector3d &globalT,
						 const Eigen::Matrix<double, 24, 3> &J_rest,
						 const SMPLModel &model)
		: smplJointIdx_(smplJointIdx), keypoint_(keypoint), camera_(camera),
		  globalR_(globalR), globalT_(globalT), J_rest_(J_rest), model_(model) {}

	template <typename T>
	bool operator()(const T *const poseParamsPtr, T *residuals) const
	{
		// Map raw pointer to Eigen vector
		Eigen::Map<const Eigen::Matrix<T, 72, 1>> poseParams(poseParamsPtr);

		// Apply SMPL Forward Kinematics
		auto poseResult = model_.applyPose<T>(poseParams, J_rest_);

		// Get specific joint position
		Eigen::Matrix<T, 3, 1> jointPos = poseResult.posedJoints.row(smplJointIdx_).transpose();

		// Apply Global Rigid Transform (R * p + t)
		Eigen::Matrix<T, 3, 1> pCam = globalR_.cast<T>() * jointPos + globalT_.cast<T>();

		// Pinhole projection
		const auto &K = camera_.intrinsics();
		const T z_inv = T(1.0) / (pCam(2) + T(1e-8));

		const T u = T(K.fx) * pCam(0) * z_inv + T(K.cx);
		const T v = T(K.fy) * pCam(1) * z_inv + T(K.cy);

		// Weighted Residuals
		const T weight = T(std::sqrt(keypoint_.score));
		residuals[0] = weight * (u - T(keypoint_.x));
		residuals[1] = weight * (v - T(keypoint_.y));

		return true;
	}

	// Index of the specific SMPL joint being optimized in this block.
	int smplJointIdx_;

	// 2D point in the image
	const Point2D &keypoint_;

	// Camera model with intrinsic parameters for 3D->2D projection.
	const CameraModel &camera_;

	// SMPL instance used to compute joints from pose parameters.
	const SMPLModel &model_;

	// Global rotation matrix (calculated in Step 1).
	Eigen::Matrix3d globalR_;

	// Global translation vector (calculated in Step 1).
	Eigen::Vector3d globalT_;

	// Pre-computed rest joint locations (based on shape) to avoid re-calculating shape inside the loop.
	// I added this to optimize performance
	Eigen::Matrix<double, 24, 3> J_rest_;
};

FittingOptimizer::FittingOptimizer(SMPLModel *smplModel_,
								   CameraModel *cameraModel_,
								   const Options &options_)
	: smplModel(smplModel_), cameraModel(cameraModel_), options(options_)
{
	poseParams.assign(72, 0.0);
	shapeParams.assign(10, 0.0);
	poseHistory.clear();
	shapeHistory.clear();
}

void FittingOptimizer::fitFrame(const Pose2D &observation)
{
	this->fitRigid(observation);
	this->fitPose(observation);
}

void FittingOptimizer::fitRigid(const Pose2D &observation)
{
	// Get Rest Joints from current shape
	Eigen::Map<Eigen::VectorXd> shapeVec(shapeParams.data(), shapeParams.size());
	auto shapeResult = smplModel->applyShape<double>(shapeVec);
	Eigen::Matrix<double, 24, 3> smplJoints = shapeResult.restJoints;

	// Initialize Global Pose via Similar Triangles
	double init_tx = 0.0, init_ty = 0.0, init_tz = 2.50;

	std::vector<double> torso3D_y, torso2D_y;
	double sum3D_x = 0, sum3D_y = 0;
	double sum2D_x = 0, sum2D_y = 0;
	int count = 0;

	for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
	{
		if (TORSO_SMPL_IDS.find(smplIdx) == TORSO_SMPL_IDS.end())
			continue;
		if (opIdx >= (int)observation.keypoints.size())
			continue;

		const Point2D &kp = observation.keypoints[opIdx];
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

	if (count >= 2)
	{
		// Estimate Depth (Z) based on torso height
		const auto [min3D, max3D] = std::minmax_element(torso3D_y.begin(), torso3D_y.end());
		const auto [min2D, max2D] = std::minmax_element(torso2D_y.begin(), torso2D_y.end());

		double h3D = *max3D - *min3D;
		double h2D = *max2D - *min2D;

		if (h2D > 1.0)
		{
			init_tz = cameraModel->intrinsics().fy * (h3D / h2D);
		}

		// Estimate Translation (X, Y) based on centroids
		double c3D_x = sum3D_x / count;
		double c3D_y = sum3D_y / count;

		double c2D_x = sum2D_x / count;
		double c2D_y = sum2D_y / count;

		double cx_cam = (c2D_x - cameraModel->intrinsics().cx) * init_tz / cameraModel->intrinsics().fx;
		double cy_cam = (c2D_y - cameraModel->intrinsics().cy) * init_tz / cameraModel->intrinsics().fy;

		init_tx = cx_cam - c3D_x;
		init_ty = cy_cam - c3D_y;
	}

	// Optimization (Torso Only)
	ceres::Problem problem;
	double pose[6] = {M_PI, 0.0, 0.0, init_tx, init_ty, init_tz};

	for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
	{
		if (TORSO_SMPL_IDS.find(smplIdx) == TORSO_SMPL_IDS.end())
			continue;
		if (opIdx >= (int)observation.keypoints.size())
			continue;

		const Point2D &kp = observation.keypoints[opIdx];
		if (kp.score < 0.2f)
			continue;

		ceres::CostFunction *cost =
			new ceres::AutoDiffCostFunction<RigidReprojectionResidual, 2, 6>(
				new RigidReprojectionResidual(
					smplJoints(smplIdx, 0), smplJoints(smplIdx, 1), smplJoints(smplIdx, 2),
					kp, *cameraModel));

		problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose);
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 50;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	// Store Optimized Rigid Transform
	Eigen::Vector3d r_vec(pose[0], pose[1], pose[2]);
	globalR_ = (r_vec.squaredNorm() > 1e-8)
				   ? Eigen::AngleAxisd(r_vec.norm(), r_vec.normalized()).toRotationMatrix()
				   : Eigen::Matrix3d::Identity();
	globalT_ = Eigen::Vector3d(pose[3], pose[4], pose[5]);
}

void FittingOptimizer::fitPose(const Pose2D &observation)
{
	// Pre-compute rest-pose joints
	Eigen::Map<Eigen::VectorXd> shapeVec(shapeParams.data(), shapeParams.size());
	auto shapeResult = smplModel->applyShape<double>(shapeVec);
	Eigen::Matrix<double, 24, 3> J_rest = shapeResult.restJoints;

	// Initialize with current pose
	std::vector<double> poseOptim = poseParams;

	ceres::Problem problem;

	for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
	{
		if (opIdx >= (int)observation.keypoints.size())
			continue;

		const Point2D &kp = observation.keypoints[opIdx];
		if (kp.score < 0.2f)
			continue;

		ceres::CostFunction *cost =
			new ceres::AutoDiffCostFunction<PoseReprojectionCost, 2, 72>(
				new PoseReprojectionCost(smplIdx, kp, *cameraModel, globalR_,
										 globalT_, J_rest, *smplModel));

		problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), poseOptim.data());
	}

	ceres::Solver::Options solverOptions;
	solverOptions.max_num_iterations = 100;
	solverOptions.linear_solver_type = ceres::DENSE_QR;
	solverOptions.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(solverOptions, &problem, &summary);

	// Update SMPL Model State
	poseParams = poseOptim;
	smplModel->setPose(poseParams);
}