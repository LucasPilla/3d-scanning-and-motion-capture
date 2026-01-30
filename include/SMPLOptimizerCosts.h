// SMPLOptimizerCosts
// -----------------
// Implements cost functors for SMPLOptimizer

#pragma once

#include "PoseDetector.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_map>

// SMPL (24) -> OpenPose BODY_25 Mapping
static const std::unordered_map<int, int> SMPL_TO_OPENPOSE = {
	{0, 8},
	{12, 1},
	{1, 12},
	{4, 13},
	{7, 14},
	{2, 9},
	{5, 10},
	{8, 11},
	{16, 5},
	{18, 6},
	{20, 7},
	{17, 2},
	{19, 3},
	{21, 4},
	// The following keypoints are not exact but provide 
	// constraints useful for estimating the right orientation.
	{15, 0},  // Nose
	{8, 24},  // Right Heel
	{7, 21},  // Left Heel
	{11, 22}, // Right Toes
	{10, 19}  // Left Toes
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
	const Point2D &keypoint_;

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
// Minimizes the error between the projected SMPL joints and the observed 2D keypoints.
struct ReprojectionCost
{
	ReprojectionCost(
		const std::vector<std::pair<int, int>> &activeMapping, 
		const std::vector<Point2D> &keypoints,
		const CameraModel &cameraModel,
		const SMPLModel &smplModel
	)
		: activeMapping_(activeMapping),
		  keypoints_(keypoints),
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

		// Apply shape and compute joints
		Eigen::Matrix<T, 24, 3> J_rest = smplModel_.getJMean().cast<T>();
		for (int i = 0; i < shapeParams.size(); ++i)
		{
			J_rest += smplModel_.getJDirs()[i].cast<T>() * shapeParams[i];
		}

		// Apply SMPL Forward Kinematics
		auto poseResult = smplModel_.applyPose<T>(poseParams, J_rest);

		for (int i = 0; i < activeMapping_.size(); ++i) 
		{
			int smplIdx = activeMapping_[i].first;
            int opIdx = activeMapping_[i].second;

			// Get specific joint position
			Eigen::Matrix<T, 3, 1> jointPos =
				poseResult.posedJoints.row(smplIdx).transpose();

			// Apply global translation
			Eigen::Matrix<T, 3, 1> pCam = jointPos + globalT;

			// Pinhole projection
			const auto &K = cameraModel_.intrinsics();
			const T z_inv = T(1.0) / (pCam(2) + T(1e-8));

			const T u = T(K.fx) * pCam(0) * z_inv + T(K.cx);
			const T v = T(K.fy) * pCam(1) * z_inv + T(K.cy);

			// Residuals
			const auto& kp = keypoints_[opIdx];
			const T conf = T(kp.score);
			residuals[i * 2] = conf * (u - T(kp.x));
			residuals[i * 2 + 1] = conf * (v - T(kp.y));
		}

		return true;
	}

	// Index of the specific SMPL joint being optimized in this block.
	const std::vector<std::pair<int, int>> &activeMapping_;

	// 2D point detected by OpenPose
	const std::vector<Point2D> &keypoints_;

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
	TemporalCost(
		double weight, 
		const std::vector<double> &prevPoseParams,
		const std::vector<double> &prevShapeParams
	): weight_(weight), prevPoseParams_(prevPoseParams), prevShapeParams_(prevShapeParams) {}

	template <typename T>
	bool operator()(const T *const poseParams, const T *const shapeParams, T *residuals) const
	{
		// Pose residuals
		for (int i = 0; i < 72; ++i)
			residuals[i] = T(weight_) * (poseParams[i] - T(prevPoseParams_[i]));
		// Shape residuals
		for (int i = 0; i < 10; ++i)
			residuals[i+72] = T(weight_) * (shapeParams[i] - T(prevShapeParams_[i]));
		return true;
	}

	double weight_;

	// Pose parameters from previos frame
	const std::vector<double> &prevPoseParams_;

	// Shape parameters from previos frame
	const std::vector<double> &prevShapeParams_;
};