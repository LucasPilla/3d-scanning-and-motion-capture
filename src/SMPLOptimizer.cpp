// SMPLOptimizer.cpp
// Implements the SMPL fitting optimization pipeline using Ceres Solver.
// This pipeline fits the SMPL 3D body model to 2D keypoints detected by OpenPose.

#include "SMPLOptimizer.h"
#include "SMPLOptimizerCosts.h"
#include "CameraModel.h"
#include "SMPLModel.h"
#include <ceres/ceres.h>
#include <cmath>
#include <limits>

SMPLOptimizer::SMPLOptimizer(
	SMPLModel *smplModel,
	CameraModel *cameraModel,
	const Options &options) : smplModel_(smplModel), cameraModel_(cameraModel), options_(options) {}

bool SMPLOptimizer::fitFrame(const std::vector<Point2D> &keypoints)
{
	// If not enougth keypoints skip frame
	int validKeypoints = 0;
	for (int i = 0; i < keypoints.size(); i++) {
		if (keypoints[i].score > 0.1) {
			validKeypoints++;
		}
	}
	
	if (validKeypoints <= 12) {
		badFrameCounter_++;
		return false;
	}

	// If warmStarting is not enabled we start optimization from scratch
	// We just use warm starting if enabled and previous frame is valid
	if (!options_.warmStarting || hasPrevFrame_ == false || badFrameCounter_ > 0)
	{
		hasPrevPrevFrame_ = false;
		hasPrevFrame_ = false;

		// Initialize pose with GMM mean
		poseParams_ = Eigen::VectorXd::Zero(72);
		Eigen::VectorXd gmmMeanPose = smplModel_->getGmmMeanPose();
		for (int i = 0; i < 69; ++i)
			poseParams_[i + 3] = gmmMeanPose(i);

		// Initialize shape with zeros
		shapeParams_ = Eigen::VectorXd::Zero(10);

		// Run first step
		fitInitialization(keypoints);
	}

	// Run second step
	bool wasSuccessfull = fitFull(keypoints);

	return wasSuccessfull;
}

void SMPLOptimizer::fitInitialization(const std::vector<Point2D> &keypoints)
{
	// Compute OpenPose joints in mean pose
	Eigen::MatrixXd openposeJoints = smplModel_->regressOpenposeRestJoints<double>(shapeParams_);

	// Default depth 
	double init_tz = 1.0;

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
	ceres::CostFunction *depthCost =
		new ceres::AutoDiffCostFunction<DepthRegularizer, 1, 3>(
			new DepthRegularizer(100.0, init_tz));

	problem.AddResidualBlock(depthCost, nullptr, translation.data());

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	options.function_tolerance = 1e-9;
	options.gradient_tolerance = 1e-5;
	options.max_num_iterations = 100;

	ceres::Solve(options, &problem, &initSummary_);

	// Store estimated transformation
	// Global translation and rotation
	globalT_ = translation;
	poseParams_[0] = rotation(0);
	poseParams_[1] = rotation(1);
	poseParams_[2] = rotation(2);
}

bool SMPLOptimizer::fitFull(const std::vector<Point2D> &keypoints)
{
	Eigen::Vector3d bestTranslation;
	Eigen::VectorXd bestShape;
	Eigen::VectorXd bestPose;
	double bestCost = std::numeric_limits<double>::max();

	for (int orientation = 0; orientation < 2; orientation++)
	{
		// Skip second orientation when using warmStarting
		if (options_.warmStarting && hasPrevFrame_ && orientation == 1)
			continue;

		// Parameters to be optimized
		Eigen::Vector3d currentTranslation = {globalT_(0), globalT_(1), globalT_(2)};
		Eigen::VectorXd currentShape = shapeParams_;
		Eigen::VectorXd currentPose = poseParams_;

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

		ceres::Problem problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = false;
		options.function_tolerance = 1e-9;
		options.gradient_tolerance = 1e-5;
		options.max_num_iterations = 25;

		// Weight variables
		double w_pose = 0.0;
        double w_shape = 0.0;
        double w_limits = 0.0;
		double w_temp_translation = 0.0;
		double w_temp_shape = 0.0;
		double w_temp_pose = 0.0;
		double w_temp_joints = 0.0;

		// Add reprojection cost (which includes the temporal residuals if enabled)

		// OpenPose keypoints ( 25 * 2 [x, y] ) + Joints ( 24 * ( 3 [acceleration] + 3 [velocity] ) )
		int num_residuals = keypoints.size() * 2; 
		if (options_.temporalRegularization && hasPrevFrame_ && hasPrevPrevFrame_)
			num_residuals += 24*3*2;

		ceres::CostFunction *cost =
			new ceres::AutoDiffCostFunction<ReprojectionCost, ceres::DYNAMIC, 3, 72, 10>(
				new ReprojectionCost(
					keypoints, 
					*cameraModel_, 
					*smplModel_, 
					&w_temp_joints, 
					options_.temporalRegularization && hasPrevFrame_ ? &prevJoints_ : nullptr,
					options_.temporalRegularization && hasPrevPrevFrame_ ? &prevPrevJoints_ : nullptr
				),
				num_residuals
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
				new GMMPosePriorCost(&w_pose, *smplModel_));

		problem.AddResidualBlock(
			gmmPosePriorCost, nullptr,
			currentPose.data());

		// Add shape prior cost
		ceres::CostFunction *shapePriorCost =
			new ceres::AutoDiffCostFunction<ShapePriorCost, 10, 10>(
				new ShapePriorCost(&w_shape));

		problem.AddResidualBlock(
			shapePriorCost, nullptr,
			currentShape.data());

		// Add joint limits cost
		ceres::CostFunction *jointLimitCost =
			new ceres::AutoDiffCostFunction<JointLimitCost, 4, 72>(
				new JointLimitCost(&w_limits));

		problem.AddResidualBlock(
			jointLimitCost,
			nullptr,
			currentPose.data());

		// Add temporal costs
		if (options_.temporalRegularization && hasPrevFrame_) {

			// Add temporal regularizer for translation
			ceres::CostFunction *transTemporalCost =
				new ceres::AutoDiffCostFunction<TemporalCost, 3, 3>(
					new TemporalCost(&w_temp_translation, prevGlobalT_));

			problem.AddResidualBlock(transTemporalCost, nullptr, currentTranslation.data());

			// Add temporal regularizer for shape
			ceres::CostFunction *shapeTemporalCost =
				new ceres::AutoDiffCostFunction<TemporalCost, 10, 10>(
					new TemporalCost(&w_temp_shape, prevShapeParams_));

			problem.AddResidualBlock(shapeTemporalCost, nullptr, currentShape.data());

			// Add temporal regularizer for pose
			ceres::CostFunction *poseTemporalCost =
				new ceres::AutoDiffCostFunction<TemporalCost, 72, 72>(
					new TemporalCost(&w_temp_pose, prevPoseParams_));

			problem.AddResidualBlock(poseTemporalCost, nullptr, currentPose.data());
		}

		// SMPLify-x weights schedule
		// Pose Prior / Shape Prior / Temporal Prior
		std::vector<std::vector<double>> weights = {
			{404.0, 100.0},
			{404.0, 50.0},
			{57.4, 10.0},
			{4.78, 5.0}
		};

		double finalCost;

		// Staged optimization
		for (int stage = 0; stage < 4; stage++)
		{
			// Update weight's variables
			w_pose = weights[stage][0];
			w_shape = weights[stage][1];
			w_limits = 1000.0;

			// Temporal weights
			// We expect joints positions to not change much between consecutive frames.
			// Ideally shape is constant, we decided to keep it variable.
			w_temp_joints = 50.0;
			w_temp_shape = 50.0;
			w_temp_pose = 0.0;
			w_temp_translation = 0.0;

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

	// Check if optimization failed
	if (
		fullSummary_.termination_type == ceres::FAILURE
	) {
		badFrameCounter_++;
		return false;
	}

	// Update optimizer state
	globalT_ = Eigen::Map<Eigen::Vector3d>(bestTranslation.data());
	poseParams_ = bestPose;
	shapeParams_ = bestShape;

	// Update SMPL model
	smplModel_->setPose(poseParams_);
	smplModel_->setShape(shapeParams_);

	// Store data for temporal residuals
	if (hasPrevFrame_) {
		hasPrevPrevFrame_ = true;
		prevPrevJoints_ = prevJoints_;
	}
	hasPrevFrame_ = true;
	badFrameCounter_ = 0;
	prevGlobalT_ = bestTranslation;
	prevPoseParams_ = bestPose;
	prevShapeParams_ = bestShape;

	// Compute final SMPL 3D joints 
	Eigen::Matrix<double, 24, 3> smplRestJoints = smplModel_->regressSmplRestJoints<double>(shapeParams_);
	auto jointTransforms = smplModel_->computeWorldTransforms<double>(poseParams_, smplRestJoints);
	for (int smplIdx = 0; smplIdx < 24; smplIdx++) 
		prevJoints_.row(smplIdx) = jointTransforms.G_trans[smplIdx];

	return true;
}