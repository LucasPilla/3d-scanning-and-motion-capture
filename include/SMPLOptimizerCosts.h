// SMPLOptimizerCosts
// -----------------
// Implements cost functors for SMPLOptimizer

#pragma once

#include "PoseDetector.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_map>

// OpenPose BODY_25 -> SMPL (24) Mapping
// Maps each OpenPose joint index to the SMPL closest joint
static const std::unordered_map<int, int> OPENPOSE_TO_SMPL = {
	{0, 15},   // Nose -> Head
	{1, 12},   // Neck -> Neck
	{2, 17},   // RShoulder -> R_Shoulder
	{3, 19},   // RElbow -> R_Elbow
	{4, 21},   // RWrist -> R_Wrist
	{5, 16},   // LShoulder -> L_Shoulder
	{6, 18},   // LElbow -> L_Elbow
	{7, 20},   // LWrist -> L_Wrist
	{8, 0},    // MidHip -> Pelvis
	{9, 2},    // RHip -> R_Hip
	{10, 5},   // RKnee -> R_Knee
	{11, 8},   // RAnkle -> R_Ankle
	{12, 1},   // LHip -> L_Hip
	{13, 4},   // LKnee -> L_Knee
	{14, 7},   // LAnkle -> L_Ankle
	{15, 15},  // REye -> Head
	{16, 15},  // LEye -> Head
	{17, 15},  // REar -> Head
	{18, 15},  // LEar -> Head
	{19, 10},  // LBigToe -> L_Foot
	{20, 10},  // LSmallToe -> L_Foot
	{21, 7},   // LHeel -> L_Ankle
	{22, 11},  // RBigToe -> R_Foot
	{23, 11},  // RSmallToe -> R_Foot
	{24, 8}    // RHeel -> R_Ankle
};

// Cost Functor: Reprojection Error (Initialization).
// Computes the reprojection error for global translation and global rotation
// for the first optimization step.
struct InitReprojectionCost
{
	InitReprojectionCost(
		float X, float Y, float Z,
		float rootX, float rootY, float rootZ,
		const Point2D &keypoint,
		const CameraModel &cameraModel)
		: X_(X), Y_(Y), Z_(Z),
		  rootX_(rootX), rootY_(rootY), rootZ_(rootZ),
		  keypoint_(keypoint),
		  cameraModel_(cameraModel) {}

	template <typename T>
	bool operator()(const T *const translation, const T *const rotation,
					T *residuals) const
	{
		// Rotation relative to SMPL root
		T centeredPoint[3];
		centeredPoint[0] = T(X_) - T(rootX_);
		centeredPoint[1] = T(Y_) - T(rootY_);
		centeredPoint[2] = T(Z_) - T(rootZ_);

		T rotatedPoint[3];
		ceres::AngleAxisRotatePoint(rotation, centeredPoint, rotatedPoint);

		// Translation
		T translatedPoint[3];
		translatedPoint[0] = rotatedPoint[0] + translation[0] + T(rootX_);
		translatedPoint[1] = rotatedPoint[1] + translation[1] + T(rootY_);
		translatedPoint[2] = rotatedPoint[2] + translation[2] + T(rootZ_);

		// Pinhole projection
		const auto &K = cameraModel_.intrinsics();
		const T z_inv = T(1.0) / (translatedPoint[2] + T(1e-8));

		const T u = T(K.fx) * (translatedPoint[0] * z_inv) + T(K.cx);
		const T v = T(K.fy) * (translatedPoint[1] * z_inv) + T(K.cy);

		// Residual
		const T weight = T(std::sqrt(keypoint_.score));
		residuals[0] = weight * (u - T(keypoint_.x));
		residuals[1] = weight * (v - T(keypoint_.y));

		return true;
	}

	// 3D center joint in SMPL
	float rootX_, rootY_, rootZ_;

	// 3D point in SMPL
	float X_, Y_, Z_;

	// 2D point detected by OpenPose
	const Point2D keypoint_;

	// Camera model with intrinsic parameters for 3D->2D projection.
	const CameraModel &cameraModel_;
};

// Cost Functor: Depth Regularization (Initialization).
// Penalizes deviation from an initial depth (Z) to prevent it to explode or vanish
struct InitDepthRegularizer
{
	InitDepthRegularizer(double init_z) : init_z_(init_z) {}

	template <typename T>
	bool operator()(const T *const translation, T *residual) const
	{
		// Residual
		residual[0] = 1e2 * (translation[2] - T(init_z_));
		return true;
	}

	double init_z_;
};

// Cost Functor: Reprojection Cost.
// Minimizes the error between the projected OpenPose joints (computed via joint regressor)
// and the observed 2D keypoints.
struct ReprojectionCost
{
	ReprojectionCost(
		const std::vector<Point2D> &keypoints,
		const CameraModel &cameraModel,
		const SMPLModel &smplModel
	)
		: keypoints_(keypoints),
		  cameraModel_(cameraModel),
		  smplModel_(smplModel) {}

	template <typename T>
	bool operator()(
		const T *const globalTPtr,
		const T *const poseParamsPtr,
		const T *const shapeParamsPtr,
		T *residuals) const
	{
		// Map raw pointer to Eigen vector
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> globalT(globalTPtr);
		Eigen::Map<const Eigen::Matrix<T, 72, 1>> poseParams(poseParamsPtr);
		Eigen::Map<const Eigen::Matrix<T, 10, 1>> shapeParams(shapeParamsPtr);

		// Compute SMPL rest joints (for forward kinematics)
		Eigen::Matrix<T, 24, 3> J_rest = smplModel_.getJMean().cast<T>();
		for (int i = 0; i < shapeParams.size(); ++i)
		{
			J_rest += smplModel_.getJDirs()[i].cast<T>() * shapeParams[i];
		}

		// Compute OpenPose rest joints using joint regressor
		Eigen::Matrix<T, 25, 3> J_openpose_rest = smplModel_.getOpenPoseJMean().cast<T>();
		for (int i = 0; i < shapeParams.size(); ++i)
		{
			J_openpose_rest += smplModel_.getOpenPoseJDirs()[i].cast<T>() * shapeParams[i];
		}

		// Apply SMPL Forward Kinematics to get global transforms
		auto poseResult = smplModel_.applyPose<T>(poseParams, J_rest);

		for (int opIdx = 0; opIdx < keypoints_.size(); opIdx++)
		{
			// Get SMPL index
			int smplIdx = OPENPOSE_TO_SMPL.at(opIdx);

			// Get OpenPose joint rest position
			Eigen::Matrix<T, 3, 1> opJointRest = J_openpose_rest.row(opIdx).transpose();

			// Get SMPL joint rest position 
			Eigen::Matrix<T, 3, 1> smplJointRest = J_rest.row(smplIdx).transpose();

			// Compute offset from SMPL joint to OpenPose joint in rest pose
			Eigen::Matrix<T, 3, 1> offset = opJointRest - smplJointRest;

			// Apply SMPL joint's global transform to the OpenPose joint
			Eigen::Matrix<T, 3, 1> jointPos =
				poseResult.G_rot[smplIdx] * offset + poseResult.G_trans[smplIdx];

			// Apply global translation
			Eigen::Matrix<T, 3, 1> pCam = jointPos + globalT;

			// Pinhole projection
			const auto &K = cameraModel_.intrinsics();
			const T z_inv = T(1.0) / (pCam(2) + T(1e-8));

			const T u = T(K.fx) * pCam(0) * z_inv + T(K.cx);
			const T v = T(K.fy) * pCam(1) * z_inv + T(K.cy);

			// Residuals
			const auto &kp = keypoints_[opIdx];
			const T conf = T(kp.score);
			residuals[opIdx * 2] = conf * (u - T(kp.x));
			residuals[opIdx * 2 + 1] = conf * (v - T(kp.y));
		}

		return true;
	}

	// 2D keypoints detected by OpenPose
	const std::vector<Point2D> keypoints_;

	// Camera model with intrinsic parameters for 3D->2D projection.
	const CameraModel &cameraModel_;

	// SMPL instance.
	const SMPLModel &smplModel_;
};

// Cost Functor: GMM Pose Prior.
// Penalizes poses that are statistically unlikely 
// based on a Gaussian Mixture Model.
struct GMMPosePriorCost
{
	GMMPosePriorCost(double weight, const SMPLModel &smplModel)
		: weight_(weight), smplModel_(smplModel) {}

	template <typename T>
	bool operator()(const T *const poseParamsPtr, T *residuals) const
	{
		// Map raw pointer to Eigen vector
		Eigen::Map<const Eigen::Matrix<T, 72, 1>> poseParams(poseParamsPtr);

		int best_gaussian = -1;
		Eigen::Matrix<T, 69, 1> best_projected_diff;
		T min_energy = T(std::numeric_limits<double>::max());

		// SMPL model data
		const Eigen::MatrixXd gmmMeans = smplModel_.getGmmMeans();
		const Eigen::MatrixXd gmmWeights = smplModel_.getGmmWeights();
		const std::vector<Eigen::MatrixXd> gmmPrecChols = smplModel_.getGmmPrecChols();

		int n_gaussians = gmmMeans.rows();
		int n_dims = gmmMeans.cols();

		for (int k = 0; k < n_gaussians; ++k)
		{
			Eigen::Matrix<T, 69, 1> mean = gmmMeans.row(k).cast<T>();

			// Compute difference to gaussian mean
			Eigen::Matrix<T, 69, 1> diff = poseParams.segment(3, 69) - mean;

			// Project: L^T * (x - mu)
			Eigen::Matrix<T, 69, 1> projected_diff = gmmPrecChols[k].cast<T>() * diff;

			// Calculate Energy
			T dist_sq = projected_diff.squaredNorm();
			T total_energy = T(0.5) * dist_sq - T(std::log(gmmWeights(k)));

			// Keep best gaussian
			if (total_energy < min_energy)
			{
				min_energy = total_energy;
				best_gaussian = k;
				best_projected_diff = projected_diff;
			}
		}

		// Residuals

		// Scaled Geometric Error
		const T scale = T(std::sqrt(0.5));
		for (int i = 0; i < 69; ++i)
		{
			residuals[i] = T(weight_) * scale * best_projected_diff(i);
		}

		// Constant statistical error
		// This ensures the optimizer fights to stay in high-probability clusters
		residuals[69] = T(weight_) * T(std::sqrt(-std::log(gmmWeights(best_gaussian))));

		return true;
	}

	double weight_;

	// SMPL instance.
	const SMPLModel &smplModel_;
};

// Cost Functor: Shape Prior.
// L2 Regularization for shape parameters (betas) to prevent extreme body deformations.
struct ShapePriorCost
{
	ShapePriorCost(double weight)
		: weight_(weight) {}

	template <typename T>
	bool operator()(const T *const shape, T *residuals) const
	{
		for (int i = 0; i < 10; ++i)
		{
			residuals[i] = T(weight_) * shape[i];
		}
		return true;
	}

	double weight_;
};

// Cost Functor: Joint Limits.
// Penalize anatomically invalid rotations for elbows and knees.
struct JointLimitCost
{
	JointLimitCost(double weight)
		: weight_(weight) {}

	template <typename T>
	bool operator()(const T *const poseParams, T *residuals) const
	{
		// Left Elbow (55), Right Elbow (58), Left Knee (12), Right Knee (15)
		residuals[0] = T(weight_) * ceres::pow(ceres::exp(-poseParams[55]), T(2));
		residuals[1] = T(weight_) * ceres::pow(ceres::exp(-poseParams[58]), T(2));
		residuals[2] = T(weight_) * ceres::pow(ceres::exp(-poseParams[12]), T(2));
		residuals[3] = T(weight_) * ceres::pow(ceres::exp(-poseParams[15]), T(2));
		return true;
	}

	double weight_;
};

// Cost Functor: Temporal Cost.
// L2 regularization penalizing deviations from the previous frame's pose parameters.
struct TemporalCost
{
	TemporalCost(double weight, const std::vector<double> &prevValues)
	: weight_(weight), prevValues_(prevValues) {}

	template <typename T>
	bool operator()(const T *const currentValues, T *residuals) const 
	{
		for (int i = 0; i < prevValues_.size(); ++i)
			residuals[i] = T(weight_) * (currentValues[i] - T(prevValues_[i]));

		return true;
	}

	double weight_;

	// Values from previos frame
	const std::vector<double> prevValues_;
};