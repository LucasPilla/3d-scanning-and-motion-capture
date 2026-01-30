// SMPLOptimizer.cpp
// Implements the SMPL fitting optimization pipeline using Ceres Solver.
// This pipeline fits the SMPL 3D body model to 2D keypoints detected by OpenPose.

#include "SMPLOptimizer.h"
#include "SMPLOptimizerCosts.h"
#include "CameraModel.h"
#include "SMPLModel.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <cmath>
#include <limits>

SMPLOptimizer::SMPLOptimizer(
	SMPLModel *smplModel,
	CameraModel *cameraModel,
	const Options &options) : smplModel_(smplModel), cameraModel_(cameraModel), options_(options) {}

void SMPLOptimizer::fitFrame(const std::vector<Point2D> &keypoints)
{
	// If OpenPose returns no detections skip frame
	if (keypoints.size() == 0)
	{
		hasPreviousFrame_ = false;
		return;
	}

	// If warmStarting is not enabled we start optimization from scratch
	// Otherwise we use current parameters, which refers to previous frame
	if (!options_.warmStarting || !hasPreviousFrame_)
	{
		// Initialize pose with GMM mean
		poseParams_.assign(72, 0.0);
		Eigen::VectorXd gmmMeanPose = smplModel_->getGmmMeanPose();
		for (int i = 0; i < 69; ++i)
			poseParams_[i + 3] = gmmMeanPose(i);

		// Initialize shape with zeros
		shapeParams_.assign(10, 0.0);

		// Run first step
		fitInitialization(keypoints);
	}

	// Run second step
	fitFull(keypoints);

	return;
}

void SMPLOptimizer::fitInitialization(const std::vector<Point2D> &keypoints)
{
	// Compute OpenPose joints from regressor
	Eigen::MatrixXd openposeJoints = smplModel_->getOpenPoseJMean();
	for (int i = 0; i < 10; ++i)
	{
		openposeJoints += smplModel_->getOpenPoseJDirs()[i] * shapeParams_[i];
	}

	// Initialize depth
	double init_tz = 2.50;

	// OpenPose torso pairs:
	// Right Shoulder (2) and Right Hip (9)
	// Left Shoulder (5) and Left Hip (12)
	std::vector<std::pair<int, int>> torsoPairs = {{2, 9}, {5, 12}};

	double totalDistance3D = 0.0;
	double totalDistance2D = 0.0;
	int validPairs = 0;

	for (const auto &pair : torsoPairs)
	{
		int shoulderOpIdx = pair.first;
		int hipOpIdx = pair.second;

		// Get OpenPose keypoints
		const Point2D &shoulderKeypoint = keypoints[shoulderOpIdx];
		const Point2D &hipKeypoint = keypoints[hipOpIdx];

		// Ignore pair with low confidence
		if (shoulderKeypoint.score < 0.1 || hipKeypoint.score < 0.1)
			continue;

		// Calculate 3D Distance using regressed OpenPose joints
		double distance3D =
			(openposeJoints.row(shoulderOpIdx) - openposeJoints.row(hipOpIdx)).norm();

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
		const auto &K = cameraModel_->intrinsics();
		init_tz = K.fy * (totalDistance3D / totalDistance2D);
	}

	// Optimization
	ceres::Problem problem;

	// Variables to be optimized
	Eigen::Vector3d translation = {0.0, 0.0, init_tz};
	Eigen::Vector3d rotation = {0.0, 0.0, 0.0};

	// OpenPose torso joint indices: RShoulder (2), RHip (9), LShoulder (5), LHip (12)
	std::vector<int> torsoOpIdxs = {2, 9, 5, 12};

	// Use OpenPose MidHip (8) as root reference, which corresponds to SMPL pelvis
	Eigen::Vector3d rootJoint = openposeJoints.row(8).transpose();

	for (int opIdx : torsoOpIdxs)
	{
		const Point2D &kp = keypoints[opIdx];

		// Get OpenPose joint position from regressor
		Eigen::Vector3d jointPos = openposeJoints.row(opIdx).transpose();

		// Add reprojection cost
		ceres::CostFunction *reprojectionCost =
			new ceres::AutoDiffCostFunction<InitReprojectionCost, 2, 3, 3>(
				new InitReprojectionCost(
					// Current joint (from OpenPose regressor)
					jointPos(0), jointPos(1), jointPos(2),
					// Root joint (OpenPose MidHip)
					rootJoint(0), rootJoint(1), rootJoint(2),
					// 2D Keypoint
					kp,
					// Camera
					*cameraModel_));

		problem.AddResidualBlock(
			reprojectionCost, nullptr, translation.data(), rotation.data());
	}

	// Add depth regularizer
	ceres::CostFunction *regularizerCost =
		new ceres::AutoDiffCostFunction<InitDepthRegularizer, 1, 3>(
			new InitDepthRegularizer(init_tz));

	problem.AddResidualBlock(regularizerCost, nullptr, translation.data());

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;

	ceres::Solve(options, &problem, &initSummary_);

	// Store estimated transformation
	globalT_ = translation;
	poseParams_[0] = rotation(0);
	poseParams_[1] = rotation(1);
	poseParams_[2] = rotation(2);
}

void SMPLOptimizer::fitFull(const std::vector<Point2D> &keypoints)
{
	std::vector<double> bestTranslation;
	std::vector<double> bestShape;
	std::vector<double> bestPose;
	double bestCost = std::numeric_limits<double>::max();

	for (int orientation = 0; orientation < 2; orientation++)
	{
		// Skip second orientation when using warmStarting
		if (hasPreviousFrame_ && options_.warmStarting && orientation == 1)
			continue;

		// Parameters to be optimized
		std::vector<double> currentTranslation = {globalT_(0), globalT_(1), globalT_(2)};
		std::vector<double> currentShape = shapeParams_;
		std::vector<double> currentPose = poseParams_;

		if (orientation == 1)
		{
			// Convert to matrix
			Eigen::Vector3d r_vec(currentPose[0], currentPose[1], currentPose[2]);
			Eigen::Matrix3d r_matrix =
				Eigen::AngleAxisd(r_vec.norm(), r_vec.normalized()).toRotationMatrix();

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
		std::vector<std::pair<double, double>> weights = {
			{4.04 * 1e2, 1e2},
			{4.04 * 1e2, 5 * 1e1},
			{57.4, 1e1},
			{4.78, 0.5 * 1e1}};

		double finalCost;

		for (int stage = 3; stage < 4; stage++)
		{
			double poseWeight = weights[stage].first;
			double shapeWeight = weights[stage].second;
			double jointLimitsWeight = 3.17 * poseWeight;

			ceres::Problem problem;

			// Add reprojection cost
			ceres::CostFunction *cost =
				new ceres::AutoDiffCostFunction<ReprojectionCost, ceres::DYNAMIC, 3, 72, 10>(
					new ReprojectionCost(keypoints, *cameraModel_, *smplModel_),
					keypoints.size() * 2 // Dynamic residual size
				);

			problem.AddResidualBlock(
				cost,
				nullptr,
				currentTranslation.data(),
				currentPose.data(),
				currentShape.data());

			// Add GMM prior cost
			ceres::CostFunction *gmmPosePriorCost =
				new ceres::AutoDiffCostFunction<GMMPosePriorCost, 70, 72>(
					new GMMPosePriorCost(poseWeight, *smplModel_));

			problem.AddResidualBlock(
				gmmPosePriorCost, nullptr,
				currentPose.data());

			// Add shape prior cost
			ceres::CostFunction *shapePriorCost =
				new ceres::AutoDiffCostFunction<ShapePriorCost, 10, 10>(
					new ShapePriorCost(shapeWeight));

			problem.AddResidualBlock(
				shapePriorCost, nullptr,
				currentShape.data());

			// Add joint limits cost
			ceres::CostFunction *jointLimitCost =
				new ceres::AutoDiffCostFunction<JointLimitCost, 4, 72>(
					new JointLimitCost(jointLimitsWeight));

			problem.AddResidualBlock(
				jointLimitCost,
				nullptr,
				currentPose.data());

			// Add temporal regularizer
			if (hasPreviousFrame_ && options_.temporalRegularization)
			{
				// Add temporal regularizer for pose
				ceres::CostFunction *poseTemporalCost =
					new ceres::AutoDiffCostFunction<TemporalCost, 72, 72>(
						new TemporalCost(100.0, prevPoseParams_));

				problem.AddResidualBlock(poseTemporalCost, nullptr, currentPose.data());

				// Add temporal regularizer for shape
				ceres::CostFunction *shapeTemporalCost =
					new ceres::AutoDiffCostFunction<TemporalCost, 10, 10>(
						new TemporalCost(100.0, prevShapeParams_));

				problem.AddResidualBlock(shapeTemporalCost, nullptr, currentShape.data());

				// Add temporal regularizer for translation
				ceres::CostFunction *translationTemporalCost =
					new ceres::AutoDiffCostFunction<TemporalCost, 3, 3>(
						new TemporalCost(100.0, prevGlobalT_));

				problem.AddResidualBlock(translationTemporalCost, nullptr, currentTranslation.data());
			}

			// Solve
			ceres::Solver::Options options;
			options.max_num_iterations = 100;
			options.linear_solver_type = ceres::DENSE_QR;
			options.minimizer_progress_to_stdout = false;

			ceres::Solve(options, &problem, &fullSummary_);

			finalCost = fullSummary_.final_cost;
		}

		if (finalCost < bestCost)
		{
			bestCost = finalCost;
			bestTranslation = currentTranslation;
			bestPose = currentPose;
			bestShape = currentShape;
		}
	}

	// Update optimizer state
	globalT_ = Eigen::Map<Eigen::Vector3d>(bestTranslation.data());
	poseParams_ = bestPose;
	shapeParams_ = bestShape;

	// Update SMPL model
	smplModel_->setPose(poseParams_);
	smplModel_->setShape(shapeParams_);

	// Save current pose for the next frame
	prevGlobalT_ = bestTranslation;
	prevPoseParams_ = bestPose;
	prevShapeParams_ = bestShape;
	hasPreviousFrame_ = true;
}