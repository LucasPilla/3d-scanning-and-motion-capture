// SMPLOptimizer.cpp
// Implements the SMPL fitting optimization pipeline.

#include "SMPLOptimizer.h"
#include "CameraModel.h"
#include "SMPLModel.h"
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <limits>
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
	{21, 4}
};

// Residual for step 1
struct InitReprojectionCost
{
	InitReprojectionCost(float X, float Y, float Z, float rootX, float rootY,
						 float rootZ, const Point2D &keypoint,
						 const CameraModel &camera)
		: X_(X), Y_(Y), Z_(Z), rootX_(rootX), rootY_(rootY), rootZ_(rootZ),
		  keypoint_(keypoint), camera_(camera) {}

	template <typename T>
	bool operator()(const T *const translation, const T *const rotation,
					T *residuals) const
	{
		T point[3] = {T(X_), T(Y_), T(Z_)};

		// Rotation relative to SMPL root
		T centeredPoint[3];
		centeredPoint[0] = T(X_) - T(rootX_);
		centeredPoint[1] = T(Y_) - T(rootY_);
		centeredPoint[2] = T(Z_) - T(rootZ_);

		// Rotation
		T rotatedPoint[3];
		ceres::AngleAxisRotatePoint(rotation, centeredPoint, rotatedPoint);

		// Translation
		T translatedPoint[3];
		translatedPoint[0] = rotatedPoint[0] + translation[0] + T(rootX_);
		translatedPoint[1] = rotatedPoint[1] + translation[1] + T(rootY_);
		translatedPoint[2] = rotatedPoint[2] + translation[2] + T(rootZ_);

		// Pinhole projection
		const auto &K = camera_.intrinsics();
		const T z_inv = T(1.0) / (translatedPoint[2] + T(1e-8));

		const T u = T(K.fx) * (translatedPoint[0] * z_inv) + T(K.cx);
		const T v = T(K.fy) * (translatedPoint[1] * z_inv) + T(K.cy);

		// Residual
		const T weight = T(std::sqrt(keypoint_.score));
		residuals[0] = weight * (u - T(keypoint_.x));
		residuals[1] = weight * (v - T(keypoint_.y));

		return true;
	}

	// SMPL center join (pelvis)
	float rootX_, rootY_, rootZ_;

	// 3D point in SMPL
	float X_, Y_, Z_;

	// 2D point in image
	const Point2D &keypoint_;

	// Camera model with intrinsic parameters for 3D->2D projection.
	const CameraModel &camera_;
};

// Residual for step 1
// This depth regularizer is required otherwise the estimated depth explodes or vanish.
struct InitDepthRegularizer
{
	InitDepthRegularizer(double init_z) : init_z_(init_z) {}

	template <typename T>
	bool operator()(const T *const translation, T *residual) const
	{
		// Residual
		residual[0] = 1e4 * (translation[2] - T(init_z_));
		return true;
	}
	double init_z_;
};

// Residual for step 2
struct ReprojectionCost
{
	ReprojectionCost(
		int smplJointIdx, const Point2D &keypoint,
		const CameraModel &camera, const SMPLModel &model,
		const Eigen::MatrixXd &J_mean, const std::vector<Eigen::MatrixXd> &J_dirs
	)
	: 	smplJointIdx_(smplJointIdx), keypoint_(keypoint), 
		camera_(camera), model_(model),
		J_mean_(J_mean), J_dirs_(J_dirs) {}

	template <typename T>
	bool operator()(const T *const globalTPtr, const T *const poseParamsPtr,
					const T *const shapeParamsPtr, T *residuals) const
	{
		// Map raw pointer to Eigen vector
		Eigen::Map<const Eigen::Matrix<T, 3, 1>> globalT(globalTPtr);
		Eigen::Map<const Eigen::Matrix<T, 72, 1>> poseParams(poseParamsPtr);
		Eigen::Map<const Eigen::Matrix<T, 10, 1>> shapeParams(shapeParamsPtr);

		// Apply shape and compute joints
    	Eigen::Matrix<T, 24, 3> J_rest = J_mean_.cast<T>(); 
		for (int i = 0; i < shapeParams.size(); ++i) {
			J_rest += J_dirs_[i].cast<T>() * shapeParams[i];
		}

		// Apply SMPL Forward Kinematics
		auto poseResult = model_.applyPose<T>(poseParams, J_rest);

		// Get specific joint position
		Eigen::Matrix<T, 3, 1> jointPos =
			poseResult.posedJoints.row(smplJointIdx_).transpose();

		// Apply global translation
		Eigen::Matrix<T, 3, 1> pCam = jointPos + globalT;

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

	// SMPL instance.
	const SMPLModel &model_;

	// Precomputed variables
	const Eigen::MatrixXd &J_mean_;
	const std::vector<Eigen::MatrixXd> &J_dirs_;
};

// Residual for step 2
struct GMMPosePriorCost
{
	GMMPosePriorCost(const SMPLModel &model)
	: model_(model) {}

	template <typename T>
	bool operator()(const T *const poseParamsPtr, T *residuals) const
	{
		// Map raw pointer to Eigen vector
		Eigen::Map<const Eigen::Matrix<T, 72, 1>> poseParams(poseParamsPtr);

        int best_gaussian = -1;
        T min_energy = T(std::numeric_limits<double>::max());

        // We need to store the projected difference vector (L * (x - mu)) 
        // for the best Gaussian to compute the final residuals.
        Eigen::Matrix<T, 69, 1> best_projected_diff;

		// SMPL model data
		const Eigen::MatrixXd gmmMeans = model_.getGmmMeans();
		const Eigen::MatrixXd gmmWeights = model_.getGmmWeights();
		const std::vector<Eigen::MatrixXd> gmmPrecChols = model_.getGmmPrecChols();

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
            if (total_energy < min_energy) {
                min_energy = total_energy;
                best_gaussian = k;
                best_projected_diff = projected_diff;
            }
        }

        // Residual
        // [0..68]: Scaled geometric error -> sqrt(0.5) * (L * (x-u))
        // [69]:    Constant statistical error -> sqrt(-log(w))

        // Scaled Geometric Error
        const T scale = T(std::sqrt(0.5));
        for (int i = 0; i < 69; ++i) {
            residuals[i] = scale * best_projected_diff(i);
        }

        // Constant statistical error 
        // This ensures the optimizer fights to stay in high-probability clusters
        residuals[69] = T(std::sqrt(-std::log(gmmWeights(best_gaussian))));

        return true;
	}

	// SMPL instance.
	const SMPLModel &model_;
};

// Residual for step 2
struct ShapePriorCost
{
	template <typename T>
	bool operator()(const T *const shape, T *residuals) const
	{
		// Skip global rotation (first 3)
		for (int i = 0; i < 10; ++i)
		{
			residuals[i] = shape[i];
		}
		return true;
	}
};

// Residual for step 2
struct JointLimitCost
{
	template <typename T>
	bool operator()(const T *const pose, T *residuals) const
	{

		const T alpha = T(10.0);

		// 55: Left Elbow
		residuals[0] = alpha * ceres::exp(pose[55]);

		// 58: Right Elbow
		residuals[1] = alpha * ceres::exp(-pose[58]);

		// 12: Left Knee
		residuals[2] = alpha * ceres::exp(-pose[12]);

		// 15: Right Knee
		residuals[3] = alpha * ceres::exp(-pose[15]);

		return true;
	}
};

SMPLOptimizer::SMPLOptimizer(SMPLModel *smplModel_, CameraModel *cameraModel_,
							 const Options &options_)
	: smplModel(smplModel_), cameraModel(cameraModel_), options(options_)
{
	poseParams.assign(72, 0.0);
	shapeParams.assign(10, 0.0);
	hasPreviousFrame_ = false;
}

void SMPLOptimizer::fitFrame(const Pose2D &observation)
{
	// If OpenPose returns no detections skip
	if (observation.keypoints.size() == 0)
		return;

	fitRigid(observation);
	fitPose(observation);

	return;
}

void SMPLOptimizer::fitRigid(const Pose2D &observation)
{
	// Get rest joints from shape parameters
	Eigen::Map<Eigen::VectorXd> shapeParamsVector(shapeParams.data(),
												  shapeParams.size());
	auto shapeResult = smplModel->applyShape<double>(shapeParamsVector);
	Eigen::Matrix<double, 24, 3> smplJoints = shapeResult.restJoints;

	// Initialize depth
	double init_tz = 2.50;

	// Pairs:
	// Right Shoulder (17) and Right Hip (2)
	// Left Shoulder (16) and Left Hip (1)
	std::vector<std::pair<int, int>> torsoPairs = {{17, 2}, {16, 1}};

	double totalDistance3D = 0.0;
	double totalDistance2D = 0.0;
	int validPairs = 0;

	for (const auto &pair : torsoPairs)
	{
		// Keypoint indexes according to SMPL
		int shoulderSmplIdx = pair.first;
		int hipSmplIdx = pair.second;

		// Keypoint indexes according to OpenPose
		int shoulderOpIdx = SMPL_TO_OPENPOSE.at(shoulderSmplIdx);
		int hipOpIdx = SMPL_TO_OPENPOSE.at(hipSmplIdx);

		// Get OpenPose keypoints
		const Point2D &shoulderKeypoint = observation.keypoints[shoulderOpIdx];
		const Point2D &hipKeypoint = observation.keypoints[hipOpIdx];

		// Ignore pair with low confidence
		if (shoulderKeypoint.score < 0.1 || hipKeypoint.score < 0.1)
			continue;

		// Calculate 3D Distance
		double distance3D =
			(smplJoints.row(shoulderSmplIdx) - smplJoints.row(hipSmplIdx)).norm();

		// Calculate 2D Distance
		double dx = shoulderKeypoint.x - hipKeypoint.x;
		double dy = shoulderKeypoint.y - hipKeypoint.y;
		double distance2D = std::sqrt(dx * dx + dy * dy);

		validPairs += 1;
		totalDistance3D += distance3D;
		totalDistance2D += distance2D;	
	}

	if (validPairs > 0)
	{
		const auto &K = cameraModel->intrinsics();
		init_tz = K.fy * (totalDistance3D / totalDistance2D);
	}

	// Optimization
	ceres::Problem problem;

	// Variables to be optimized
	std::vector<double> translation = {0.0, 0.0, init_tz};
	std::vector<double> rotation = {0.0, 0.0, 0.0};

	// Torso keypoint indexes
	std::vector<int> torsoIdxs = {17, 2, 16, 1};

	for (int smplIdx : torsoIdxs)
	{
		int opIdx = SMPL_TO_OPENPOSE.at(smplIdx);
		const Point2D &kp = observation.keypoints[opIdx];

		ceres::CostFunction *projectionCost =
			new ceres::AutoDiffCostFunction<InitReprojectionCost, 2, 3, 3>(
				new InitReprojectionCost(
					// Current joint
					smplJoints(smplIdx, 0), smplJoints(smplIdx, 1), smplJoints(smplIdx, 2),
					// Root joint
					smplJoints(0, 0), smplJoints(0, 1), smplJoints(0, 2),
					// 2D Keypoint
					kp,
					// Camera
					*cameraModel)
				);

		// Add reprojection residual
		problem.AddResidualBlock(
			projectionCost, nullptr, translation.data(), rotation.data()
		);
	}

	// Add depth regularizer residual
	ceres::CostFunction *regularizerCost =
		new ceres::AutoDiffCostFunction<InitDepthRegularizer, 1, 3>(
			new InitDepthRegularizer(init_tz));
	problem.AddResidualBlock(regularizerCost, nullptr, translation.data());

	ceres::Solver::Summary summary;

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solve(options, &problem, &summary);

	// Store estimated transformation
	globalT_ = Eigen::Vector3d(translation[0], translation[1], translation[2]);
	poseParams[0] = rotation[0];
	poseParams[1] = rotation[1];
	poseParams[2] = rotation[2];
	smplModel->setPose(poseParams);

	// // Store diagnostics
	// lastFitRigidCost_ = summary.final_cost;
	// lastFitRigidIters_ = summary.num_successful_steps;
}

void SMPLOptimizer::fitPose(const Pose2D &observation)
{
	// SMPL data 
	const Eigen::MatrixXd& T_mean = smplModel->getTemplateVertices(); 
    const Eigen::MatrixXd& J_reg  = smplModel->getJointRegressor();     
    const Eigen::MatrixXd& S_dirs = smplModel->getShapeBlendShapes();

	// Precompute J_mean
	const Eigen::MatrixXd J_mean =  J_reg * T_mean;

	// Precompute J_dirs
	std::vector<Eigen::MatrixXd> J_dirs(10);
    for (int i = 0; i < 10; ++i) {
        // Get vector coresponding to beta[i]
        Eigen::VectorXd shape_vector = S_dirs.col(i); 
        
        // Reshape  (20670, 1) to (6890, 3)
        Eigen::Map<const Eigen::Matrix<double, 6890, 3, Eigen::RowMajor>> 
    		shape_vector_reshaped(shape_vector.data());

		// Compute J_dirs
        J_dirs[i] = J_reg * shape_vector_reshaped;
    }

	Eigen::Vector3d bestTranslation;
	std::vector<double> bestShape;
	std::vector<double> bestPose;
	double bestCost = std::numeric_limits<double>::max();

	for (int orientation = 0; orientation < 2; orientation++)
	{
		// Parameters to be optimized
		Eigen::Vector3d currentTranslation = globalT_;
		std::vector<double> currentShape = shapeParams;
		std::vector<double> currentPose = poseParams;

		if (orientation == 1)
		{
			// Convert to matrix
			Eigen::Vector3d r_vec(currentPose[0], currentPose[1], currentPose[2]);
			Eigen::Matrix3d r_matrix =
				Eigen::AngleAxisd(r_vec.norm(), r_vec.normalized())
					.toRotationMatrix();

			// Invert orientation
			Eigen::Matrix3d Ry;
			Ry << -1, 0, 0, 0, 1, 0, 0, 0, -1;
			r_matrix = r_matrix * Ry;

			// Convert back to parameters
			Eigen::AngleAxisd new_r(r_matrix);
			Eigen::Vector3d r_parameters = new_r.angle() * new_r.axis();
			currentPose[0] = r_parameters[0];
			currentPose[1] = r_parameters[1];
			currentPose[2] = r_parameters[2];
		}

		// Official SMPLify weights schedule
		std::vector<std::pair<double, double>> weights = {{4.04 * 1e2, 1e2},
														  {4.04 * 1e2, 5 * 1e1},
														  {57.4, 1e1},
														  {4.78, 0.5 * 1e1}};

		double finalCost;

		for (int stage = 0; stage < 4; stage++)
		{

			double poseWeight = weights[stage].first;
			double shapeWeight = weights[stage].second;
			double jointLimitsWeight = 0.317 * poseWeight;

			ceres::Problem problem;

			// Add reprojection cost for each keypoint
			for (const auto &[smplIdx, opIdx] : SMPL_TO_OPENPOSE)
			{
				const Point2D &kp = observation.keypoints[opIdx];

				if (kp.score < 0.2)
				{
					continue;
				}

				ceres::CostFunction *cost =
					new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 3, 72, 10>(
						new ReprojectionCost(smplIdx, kp, *cameraModel, *smplModel, J_mean, J_dirs));

				problem.AddResidualBlock(cost, nullptr,
										 currentTranslation.data(), currentPose.data(),
										 currentShape.data());
			}

			// Add pose prior cost
			ceres::CostFunction *gmmPosePriorCost =
				new ceres::AutoDiffCostFunction<GMMPosePriorCost, 70, 72>(
					new GMMPosePriorCost(*smplModel));
			problem.AddResidualBlock(gmmPosePriorCost,
									 new ceres::ScaledLoss(nullptr,
														   poseWeight*poseWeight,
														   ceres::TAKE_OWNERSHIP),
									 currentPose.data());

			// Add shape prior cost
			ceres::CostFunction *shapePriorCost =
				new ceres::AutoDiffCostFunction<ShapePriorCost, 10, 10>(
					new ShapePriorCost());
			problem.AddResidualBlock(shapePriorCost,
									 new ceres::ScaledLoss(nullptr,
														   shapeWeight*shapeWeight,
														   ceres::TAKE_OWNERSHIP),
									 currentShape.data());

			// Add joint limits cost
			ceres::CostFunction *jointLimitCost =
				new ceres::AutoDiffCostFunction<JointLimitCost, 4, 72>(
					new JointLimitCost());
			problem.AddResidualBlock(
				jointLimitCost,
				new ceres::ScaledLoss(nullptr, jointLimitsWeight*jointLimitsWeight,
									  ceres::TAKE_OWNERSHIP),
				currentPose.data());

			// Solve
			ceres::Solver::Options solverOptions;
			solverOptions.max_num_iterations = 100;
			solverOptions.linear_solver_type = ceres::DENSE_QR;
			solverOptions.minimizer_progress_to_stdout = false;

			ceres::Solver::Summary summary;
			ceres::Solve(solverOptions, &problem, &summary);

			finalCost = summary.final_cost;
		}

		if (finalCost < bestCost)
		{
			bestCost = finalCost;
			bestTranslation = currentTranslation;
			bestPose = currentPose;
		}
	}

	// Update optimizer state
	globalT_ = bestTranslation;
	poseParams = bestPose;

	// Update SMPL model
	smplModel->setPose(poseParams);
}