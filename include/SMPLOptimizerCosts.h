// SMPLOptimizerCosts
// -----------------
// Implements cost functors for SMPLOptimizer

#pragma once

#include "PoseDetector.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// OpenPose BODY_25 -> SMPL (24) Mapping
// Index = OpenPose joint index, Value = SMPL closest joint
static const std::vector<int> OPENPOSE_TO_SMPL = {
	15, // 0  Nose -> Head
	12, // 1  Neck -> Neck
	17, // 2  RShoulder -> R_Shoulder
	19, // 3  RElbow -> R_Elbow
	21, // 4  RWrist -> R_Wrist
	16, // 5  LShoulder -> L_Shoulder
	18, // 6  LElbow -> L_Elbow
	20, // 7  LWrist -> L_Wrist
	0,  // 8  MidHip -> Pelvis
	2,  // 9  RHip -> R_Hip
	5,  // 10 RKnee -> R_Knee
	8,  // 11 RAnkle -> R_Ankle
	1,  // 12 LHip -> L_Hip
	4,  // 13 LKnee -> L_Knee
	7,  // 14 LAnkle -> L_Ankle
	15, // 15 REye -> Head
	15, // 16 LEye -> Head
	15, // 17 REar -> Head
	15, // 18 LEar -> Head
	10, // 19 LBigToe -> L_Foot
	10, // 20 LSmallToe -> L_Foot
	7,  // 21 LHeel -> L_Ankle
	11, // 22 RBigToe -> R_Foot
	11, // 23 RSmallToe -> R_Foot
	8   // 24 RHeel -> R_Ankle
};

// This mapping is only used for weighting the velocity term of temporal residuals. 
// Index = SMPL Joint Index (0-23)
// Value = OpenPose BODY_25 Index 
static constexpr int SMPL_TO_OPENPOSE[24] = {
    8,  // 0  Pelvis      -> MidHip (8)
    12, // 1  L_Hip       -> LHip (12)
    9,  // 2  R_Hip       -> RHip (9)
    8,  // 3  Spine1      -> PROXY: MidHip (8) (Lower spine visibility)
    13, // 4  L_Knee      -> LKnee (13)
    10, // 5  R_Knee      -> RKnee (10)
    1,  // 6  Spine2      -> PROXY: Neck (1) (Upper spine visibility)
    14, // 7  L_Ankle     -> LAnkle (14)
    11, // 8  R_Ankle     -> RAnkle (11)
    1,  // 9  Spine3      -> PROXY: Neck (1) (Chest visibility)
    19, // 10 L_Foot      -> LBigToe (19)
    22, // 11 R_Foot      -> RBigToe (22)
    1,  // 12 Neck        -> Neck (1)
    5,  // 13 L_Collar    -> PROXY: LShoulder (5) (Collar follows shoulder)
    2,  // 14 R_Collar    -> PROXY: RShoulder (2) (Collar follows shoulder)
    0,  // 15 Head        -> Nose (0)
    5,  // 16 L_Shoulder  -> LShoulder (5)
    2,  // 17 R_Shoulder  -> RShoulder (2)
    6,  // 18 L_Elbow     -> LElbow (6)
    3,  // 19 R_Elbow     -> RElbow (3)
    7,  // 20 L_Wrist     -> LWrist (7)
    4,  // 21 R_Wrist     -> RWrist (4)
    7,  // 22 L_Hand      -> PROXY: LWrist (7) (Hand follows wrist)
    4   // 23 R_Hand      -> PROXY: RWrist (4) (Hand follows wrist)
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
		const T factor = T(1000.0 / cameraModel_.getFrameHeight());
		const T confidence = T(keypoint_.score);
		residuals[0] = factor * confidence * (u - T(keypoint_.x));
		residuals[1] = factor * confidence * (v - T(keypoint_.y));

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

// Cost Functor: Depth Regularization.
// Penalizes deviation from an initial depth (Z) to prevent it to explode or vanish
struct DepthRegularizer
{
	DepthRegularizer(double weight, double init_z) 
	: weight_(weight), init_z_(init_z) {}

	template <typename T>
	bool operator()(const T *const translation, T *residual) const
	{
		// Residual
		residual[0] = T(weight_) * (translation[2] - T(init_z_));
		return true;
	}

	double weight_;
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
		const SMPLModel &smplModel,
		const double *temporalWeight,
		const Eigen::Matrix<double, 24, 3> *prevJoints,
		const Eigen::Matrix<double, 24, 3> *prevPrevJoints
	) : 
		keypoints_(keypoints),
		cameraModel_(cameraModel),
		smplModel_(smplModel),
		temporalWeight_(temporalWeight),
		prevJoints_(prevJoints),
		prevPrevJoints_(prevPrevJoints)
	{}

	template <typename T>
	bool operator()(
		const T *const globalTPtr,
		const T *const poseParamsPtr,
		const T *const shapeParamsPtr,
		T *residuals) const
	{
		int residualOffset = 0;

		// Map raw pointer to Eigen vector
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> globalT(globalTPtr);
		Eigen::Map<const Eigen::Matrix<T, 72, 1>> poseParams(poseParamsPtr);
		Eigen::Map<const Eigen::Matrix<T, 10, 1>> shapeParams(shapeParamsPtr);

		// Compute SMPL rest joints 
		Eigen::Matrix<T, 24, 3> J_rest = smplModel_.regressSmplRestJoints<T>(shapeParams);

		// Compute OpenPose rest joints using joint regressor
		Eigen::Matrix<T, 25, 3> J_openpose_rest = smplModel_.regressOpenposeRestJoints<T>(shapeParams);

		// Compute global transforms for joints
		auto jointTransforms = smplModel_.computeWorldTransforms<T>(poseParams, J_rest);

		for (int opIdx = 0; opIdx < keypoints_.size(); opIdx++)
		{
			// Get SMPL index
			int smplIdx = OPENPOSE_TO_SMPL[opIdx];

			// Get OpenPose joint rest position
			Eigen::Matrix<T, 3, 1> opJointRest = J_openpose_rest.row(opIdx).transpose();

			// Get SMPL joint rest position 
			Eigen::Matrix<T, 3, 1> smplJointRest = J_rest.row(smplIdx).transpose();

			// Compute offset from SMPL joint to OpenPose joint in rest pose
			Eigen::Matrix<T, 3, 1> offset = opJointRest - smplJointRest;

			// Apply SMPL joint's global transform to the OpenPose joint
			Eigen::Matrix<T, 3, 1> jointPos =
				jointTransforms.G_rot[smplIdx] * offset + jointTransforms.G_trans[smplIdx];

			// Apply global translation
			Eigen::Matrix<T, 3, 1> pCam = jointPos + globalT;

			// Pinhole projection
			const auto &K = cameraModel_.intrinsics();
			const T z_inv = T(1.0) / (pCam(2) + T(1e-8));

			const T u = T(K.fx) * pCam(0) * z_inv + T(K.cx);
			const T v = T(K.fy) * pCam(1) * z_inv + T(K.cy);

			// Compute residuals
			const auto &kp = keypoints_[opIdx];
            T err_x = u - T(kp.x);
            T err_y = v - T(kp.y);

			// Weight by keypoint confidence
            T confidence = T(kp.score);

			// Weight according to frame height
			T factor = T(1000.0 / cameraModel_.getFrameHeight());

            residuals[residualOffset++] = factor * confidence * err_x;
            residuals[residualOffset++] = factor * confidence * err_y;
        }

		// Add temporal residuals based on joints "acceleration" and "velocity".
		// We consider position w.r.t. root joint, not world coordinates.
		// We keep this within ReprojectionCost to avoid recomputing forward kinematics.
		if (prevJoints_ && prevPrevJoints_) {
            T w_acc = T(*temporalWeight_);
            for (int smplIdx = 0; smplIdx < 24; smplIdx++) {
				// Get history
				Eigen::Matrix<T, 3, 1> prevPrevJoint = prevPrevJoints_->row(smplIdx).transpose().cast<T>();
				Eigen::Matrix<T, 3, 1> prevJoint = prevJoints_->row(smplIdx).transpose().cast<T>();
				Eigen::Matrix<T, 3, 1> currentJoint = jointTransforms.G_trans[smplIdx];

				// Compute "derivatives"
				Eigen::Matrix<T, 3, 1> prevVelocity = prevJoint - prevPrevJoint;
				Eigen::Matrix<T, 3, 1> velocity = currentJoint - prevJoint;
				Eigen::Matrix<T, 3, 1> acceleration = velocity - prevVelocity;

				// Acceleration term 
				residuals[residualOffset++] = w_acc * acceleration.x();
				residuals[residualOffset++] = w_acc * acceleration.y();
				residuals[residualOffset++] = w_acc * acceleration.z();

				// Velocity term (friction to stop movement when joint is not visible)
				int opIdx = SMPL_TO_OPENPOSE[smplIdx];
				Point2D kp = keypoints_[opIdx];
				T w_vel = kp.score > 0.1 ? T(0.1) * w_acc : T(0.5) * w_acc;
				residuals[residualOffset++] = w_vel * velocity.x();
				residuals[residualOffset++] = w_vel * velocity.y();
				residuals[residualOffset++] = w_vel * velocity.z();
        	}
        }

		return true;
	}

	// 2D keypoints detected by OpenPose
	const std::vector<Point2D> keypoints_;

	// Camera model with intrinsic parameters for 3D->2D projection.
	const CameraModel &cameraModel_;

	// SMPL instance.
	const SMPLModel &smplModel_;

	// Weight for temporal residuals
	const double *temporalWeight_;

	// Previous joints for temporal residuals
	const Eigen::Matrix<double, 24, 3> *prevJoints_;
	const Eigen::Matrix<double, 24, 3> *prevPrevJoints_;
};

// Cost Functor: GMM Pose Prior.
// Penalizes poses that are statistically unlikely 
// based on a Gaussian Mixture Model.
struct GMMPosePriorCost
{
	GMMPosePriorCost(const double *weight, const SMPLModel &smplModel)
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
		const Eigen::MatrixXd &gmmMeans = smplModel_.getGmmMeans();
		const Eigen::MatrixXd &gmmWeights = smplModel_.getGmmWeights();
		const std::vector<Eigen::MatrixXd> &gmmPrecChols = smplModel_.getGmmPrecChols();

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
		const T weight = T(*weight_);
		for (int i = 0; i < 69; ++i)
		{
			residuals[i] = weight * scale * best_projected_diff(i);
		}

		// Constant statistical error
		// This ensures the optimizer fights to stay in high-probability clusters
		residuals[69] = weight * T(std::sqrt(-std::log(gmmWeights(best_gaussian))));

		return true;
	}

	// Reference to weight variable so this weight
	// can be updated without recreating the residual block.
	const double *weight_;

	// SMPL instance.
	const SMPLModel &smplModel_;
};

// Cost Functor: Shape Prior.
// L2 Regularization for shape parameters (betas) to prevent extreme body deformations.
struct ShapePriorCost
{
	ShapePriorCost(const double *weight)
		: weight_(weight) {}

	template <typename T>
	bool operator()(const T *const shape, T *residuals) const
	{
		T weight = T(*weight_);
		for (int i = 0; i < 10; ++i)
			residuals[i] = weight * shape[i];
		return true;
	}

	// Reference to weight variable so this weight
	// can be updated without recreating the residual block.
	const double *weight_;
};

// Cost Functor: Joint Limits.
// Penalize anatomically invalid rotations for elbows and knees.
struct JointLimitCost
{
	JointLimitCost(const double *weight)
		: weight_(weight) {}

	template <typename T>
	bool operator()(const T *const poseParams, T *residuals) const
	{
		// Left Elbow (55), Right Elbow (58), Left Knee (12), Right Knee (15)
		T weight = T(*weight_);
		residuals[0] = weight * ceres::fmax(poseParams[55], T(0.0));
		residuals[1] = weight * ceres::fmax(-poseParams[58], T(0.0));
		residuals[2] = weight * ceres::fmax(-poseParams[12], T(0.0));
		residuals[3] = weight * ceres::fmax(-poseParams[15], T(0.0));
		return true;
	}

	// Reference to weight variable so this weight
	// can be updated without recreating the residual block.
	const double *weight_;
};

// Cost Functor: Temporal Cost.
// L2 regularization penalizing deviations from the previous frame's pose parameters.
struct TemporalCost
{
	TemporalCost(double *weight, const Eigen::VectorXd &prevValues)
	: weight_(weight), prevValues_(prevValues) {}

	template <typename T>
	bool operator()(const T *const currentValues, T *residuals) const 
	{
		for (int i = 0; i < prevValues_.size(); ++i)
			residuals[i] = T(*weight_) * (currentValues[i] - T(prevValues_(i)));

		return true;
	}

	double *weight_;

	// Values from previos frame
	const Eigen::VectorXd prevValues_;
};

