// SMPLModel.cpp
// --------
// Implements the SMPLModel interface defined in SMPLModel.h.

#include "SMPLModel.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using nlohmann::json;

bool SMPLMesh::save(const std::string &path) const
{

	std::ofstream outFile(path);
	if (!outFile.is_open())
	{
		std::cerr << "Error: Could not open file for writing: " << path
				  << std::endl;
		return false;
	}

	// 1. Save vertices (v x y z)
	for (const auto &vertex : vertices)
	{
		if (vertex.allFinite())
		{
			outFile << "v " << vertex.x() << " " << vertex.y() << " " << vertex.z()
					<< "\n";
		}
		else
		{
			outFile << "v 0.0 0.0 0.0\n";
		}
	}

	// 2. Save faces (f v1 v2 v3) - OBJ is 1-indexed!
	for (const auto &face : faces)
	{
		outFile << "f " << face[0] + 1 << " " << face[1] + 1 << " " << face[2] + 1
				<< "\n";
	}

	outFile.close();
	return true;
}

bool SMPLModel::loadFromJson(const std::string &jsonPath)
{

	std::ifstream in(jsonPath);
	if (!in.is_open())
	{
		std::cerr << "SMPLModel::loadFromJson - cannot open file: " << jsonPath
				  << "\n";
		return false;
	}

	json j;
	try
	{
		in >> j;
	}
	catch (const std::exception &e)
	{
		std::cerr << "SMPLModel::loadFromJson - JSON parse error: " << e.what()
				  << "\n";
		return false;
	}

	try
	{
		// -------- vertices_template (N, 3) --------
		const auto &vt = j.at("vertices_template");
		const int numVertices = static_cast<int>(vt.size());
		if (numVertices == 0 || vt.at(0).size() != 3)
		{
			std::cerr
				<< "SMPLModel::loadFromJson - invalid vertices_template shape\n";
			return false;
		}

		templateVertices_.resize(numVertices, 3);
		for (int i = 0; i < numVertices; ++i)
		{
			for (int c = 0; c < 3; ++c)
			{
				templateVertices_(i, c) = static_cast<double>(vt[i][c]);
			}
		}

		// -------- face_indices (F, 3) --------
		const auto &faces = j.at("face_indices");
		const int numFaces = static_cast<int>(faces.size());
		faces_.resize(numFaces, 3);
		for (int i = 0; i < numFaces; ++i)
		{
			for (int c = 0; c < 3; ++c)
			{
				// preprocess.py stores 1-based indices; convert to 0-based for C++
				int idx = static_cast<int>(faces[i][c]) - 1;
				faces_(i, c) = idx;
			}
		}

		// -------- shape_blend_shapes (N, 3, numShapeCoeffs) --------
		const auto &shapeBs = j.at("shape_blend_shapes");
		const int numShapeVerts = static_cast<int>(shapeBs.size()); // should be N
		const int numShapeComps =
			static_cast<int>(shapeBs[0].size()); // should be 3
		const int numShapeCoeffs =
			static_cast<int>(shapeBs[0][0].size()); // e.g. 10

		if (numShapeVerts != numVertices || numShapeComps != 3)
		{
			std::cerr
				<< "SMPLModel::loadFromJson - invalid shape_blend_shapes shape\n";
			return false;
		}

		shapeBlendShapes_.resize(numVertices * 3, numShapeCoeffs);
		for (int v = 0; v < numVertices; ++v)
		{
			for (int c = 0; c < 3; ++c)
			{
				int row = v * 3 + c;
				for (int k = 0; k < numShapeCoeffs; ++k)
				{
					shapeBlendShapes_(row, k) = static_cast<double>(shapeBs[v][c][k]);
				}
			}
		}

		// NOTE: In SMPL, posedirs/pose_blend_shapes is stored as a 3D tensor
		// (numVertices, 3, numPoseCoeffs). We flatten it row-wise to a
		// (numVertices * 3, numPoseCoeffs) matrix here.
		// -------- pose_blend_shapes (N, 3, numPoseCoeffs) --------
		const auto &poseBs = j.at("pose_blend_shapes");
		const int numPoseVerts = static_cast<int>(poseBs.size());
		const int numPoseComps = static_cast<int>(poseBs[0].size());
		const int numPoseCoeffs = static_cast<int>(poseBs[0][0].size()); // e.g. 207

		if (numPoseVerts != numVertices || numPoseComps != 3)
		{
			std::cerr
				<< "SMPLModel::loadFromJson - invalid pose_blend_shapes shape\n";
			return false;
		}

		poseBlendShapes_.resize(numVertices * 3, numPoseCoeffs);
		for (int v = 0; v < numVertices; ++v)
		{
			for (int c = 0; c < 3; ++c)
			{
				int row = v * 3 + c;
				for (int k = 0; k < numPoseCoeffs; ++k)
				{
					poseBlendShapes_(row, k) = static_cast<double>(poseBs[v][c][k]);
				}
			}
		}

		// -------- joint_regressor (numJoints, N) --------
		const auto &jr = j.at("joint_regressor");
		const int numJoints = static_cast<int>(jr.size());
		if (numJoints == 0 || static_cast<int>(jr[0].size()) != numVertices)
		{
			std::cerr << "SMPLModel::loadFromJson - invalid joint_regressor shape\n";
			return false;
		}

		jointRegressor_.resize(numJoints, numVertices);
		for (int jIdx = 0; jIdx < numJoints; ++jIdx)
		{
			for (int v = 0; v < numVertices; ++v)
			{
				jointRegressor_(jIdx, v) = static_cast<double>(jr[jIdx][v]);
			}
		}

		// -------- kinematic_tree (2, numJoints) --------
		const auto &kt = j.at("kinematic_tree");
		const int ktRows = static_cast<int>(kt.size()); // Expected: 2
		const int ktCols =
			static_cast<int>(kt[0].size()); // Expected: numJoints (24)

		kinematicTree_.resize(ktRows, ktCols);
		for (int r = 0; r < ktRows; ++r)
		{
			for (int c = 0; c < ktCols; ++c)
			{
				kinematicTree_(r, c) = static_cast<int>(kt[r][c]);
			}
		}

		// -------- weights (N, numJoints) --------
		const auto &w = j.at("weights");
		if (static_cast<int>(w.size()) != numVertices)
		{
			std::cerr << "SMPLModel::loadFromJson - invalid weights shape\n";
			return false;
		}

		weights_.resize(numVertices, numJoints);
		for (int v = 0; v < numVertices; ++v)
		{
			for (int jIdx = 0; jIdx < numJoints; ++jIdx)
			{
				weights_(v, jIdx) = static_cast<double>(w[v][jIdx]);
			}
		}

        // -------- gmm_means --------
        if (j.contains("gmm_means"))
        {
            const auto &gmm_means = j.at("gmm_means");
            int gmmMeansRows = static_cast<int>(gmm_means.size());
            int gmmMeansCols = static_cast<int>(gmm_means[0].size());
            gmmMeans_.resize(gmmMeansRows, gmmMeansCols);
            for (int i = 0; i < gmmMeansRows; ++i)
                for (int k = 0; k < gmmMeansCols; ++k)
                    gmmMeans_(i, k) = static_cast<double>(gmm_means[i][k]);
        }

        // -------- gmm_covars --------
        // JSON is 3D (8, 69, 69). We flatten the last two dimensions to fit Eigen Matrix (8, 4761).
        if (j.contains("gmm_covars"))
        {
            const auto &gmm_covars = j.at("gmm_covars");
            int gmmCovarsRows = static_cast<int>(gmm_covars.size());
            int gmmCovarsDim1 = static_cast<int>(gmm_covars[0].size());
            int gmmCovarsDim2 = static_cast<int>(gmm_covars[0][0].size());
            
            gmmCovars_.resize(gmmCovarsRows, gmmCovarsDim1 * gmmCovarsDim2);
            
            for (int i = 0; i < gmmCovarsRows; ++i)
            {
                for (int r = 0; r < gmmCovarsDim1; ++r)
                {
                    for (int c = 0; c < gmmCovarsDim2; ++c)
                    {
                        int flatIdx = r * gmmCovarsDim2 + c;
                        gmmCovars_(i, flatIdx) = static_cast<double>(gmm_covars[i][r][c]);
                    }
                }
            }
        }

        // -------- gmm_weights --------
        if (j.contains("gmm_weights"))
        {
            const auto &gmm_weights = j.at("gmm_weights");
            int gmmWeightsRows = static_cast<int>(gmm_weights.size());
            gmmWeights_.resize(gmmWeightsRows, 1);
            for (int i = 0; i < gmmWeightsRows; ++i)
                gmmWeights_(i, 0) = static_cast<double>(gmm_weights[i]);
        }

        // -------- capsule_v2lens --------
        if (j.contains("capsule_v2lens"))
        {
            const auto &capsule_v2lens = j.at("capsule_v2lens");
            int capsuleV2lensRows = static_cast<int>(capsule_v2lens.size());
            int capsuleV2lensCols = static_cast<int>(capsule_v2lens[0].size());
            capsuleV2Lens_.resize(capsuleV2lensRows, capsuleV2lensCols);
            for (int i = 0; i < capsuleV2lensRows; ++i)
                for (int k = 0; k < capsuleV2lensCols; ++k)
                    capsuleV2Lens_(i, k) = static_cast<double>(capsule_v2lens[i][k]);
        }

        // -------- capsule_betas2lens --------
        if (j.contains("capsule_betas2lens"))
        {
            const auto &capsule_betas2lens = j.at("capsule_betas2lens");
            int capsuleBetas2lensRows = static_cast<int>(capsule_betas2lens.size());
            int capsuleBetas2lensCols = static_cast<int>(capsule_betas2lens[0].size());
            capsuleBetas2Lens_.resize(capsuleBetas2lensRows, capsuleBetas2lensCols);
            for (int i = 0; i < capsuleBetas2lensRows; ++i)
                for (int k = 0; k < capsuleBetas2lensCols; ++k)
                    capsuleBetas2Lens_(i, k) = static_cast<double>(capsule_betas2lens[i][k]);
        }

        // -------- capsule_v2rads --------
        if (j.contains("capsule_v2rads"))
        {
            const auto &capsule_v2rads = j.at("capsule_v2rads");
            int capsuleV2radsRows = static_cast<int>(capsule_v2rads.size());
            int capsuleV2radsCols = static_cast<int>(capsule_v2rads[0].size());
            capsuleV2Rads_.resize(capsuleV2radsRows, capsuleV2radsCols);
            for (int i = 0; i < capsuleV2radsRows; ++i)
                for (int k = 0; k < capsuleV2radsCols; ++k)
                    capsuleV2Rads_(i, k) = static_cast<double>(capsule_v2rads[i][k]);
        }

        // -------- capsule_betas2rads --------
        if (j.contains("capsule_betas2rads"))
        {
            const auto &capsule_betas2rads = j.at("capsule_betas2rads");
            int capsuleBetas2radsRows = static_cast<int>(capsule_betas2rads.size());
            int capsuleBetas2radsCols = static_cast<int>(capsule_betas2rads[0].size());
            capsuleBetas2Rads_.resize(capsuleBetas2radsRows, capsuleBetas2radsCols);
            for (int i = 0; i < capsuleBetas2radsRows; ++i)
                for (int k = 0; k < capsuleBetas2radsCols; ++k)
                    capsuleBetas2Rads_(i, k) = static_cast<double>(capsule_betas2rads[i][k]);
        }
	}
	catch (const std::exception &e)
	{
		std::cerr << "SMPLModel::loadFromJson - error while reading arrays: "
				  << e.what() << "\n";
		return false;
	}

	// Initialize parameter vectors
	poseParams_ = Eigen::VectorXd::Zero(3 * jointRegressor_.rows());
	shapeParams_ = Eigen::VectorXd::Zero(shapeBlendShapes_.cols());

	std::cout << "SMPLModel::loadFromJson - loaded model from " << jsonPath
			  << "\n";

	return true;
}

void SMPLModel::setPose(const std::vector<double> &poseParams)
{
	poseParams_.resize(static_cast<int>(poseParams.size()));
	for (int i = 0; i < static_cast<int>(poseParams.size()); ++i)
	{
		poseParams_[i] = static_cast<double>(poseParams[i]);
	}
}

void SMPLModel::setShape(const std::vector<double> &shapeParams)
{
	shapeParams_.resize(static_cast<int>(shapeParams.size()));
	for (int i = 0; i < static_cast<int>(shapeParams.size()); ++i)
	{
		shapeParams_[i] = static_cast<double>(shapeParams[i]);
	}
}

SMPLMesh SMPLModel::computeMesh() const
{
	SMPLMesh mesh;

	const int N = templateVertices_.rows();
	const int numJoints = jointRegressor_.rows();

	// 1. Apply shape: get v_shaped and J_rest
	Eigen::Matrix<double, 10, 1> beta;
	for (int i = 0; i < 10; ++i)
	{
		beta(i) = shapeParams_(i);
	}
	auto shapeResult = applyShape<double>(beta);
	const auto &v_shaped = shapeResult.shapedVertices;
	const auto &J = shapeResult.restJoints;

	// 2. Apply pose: get rotations, G_rot, G_trans
	Eigen::Matrix<double, 72, 1> theta;
	for (int i = 0; i < 72; ++i)
	{
		theta(i) = poseParams_(i);
	}
	auto poseResult = applyPose<double>(theta, J);

	// 3. Pose blend shapes (uses rotations from poseResult)
	Eigen::VectorXd pose_map((numJoints - 1) * 9);
	for (int i = 1; i < numJoints; ++i)
	{
		Eigen::Matrix3d diff =
			poseResult.rotations[i] - Eigen::Matrix3d::Identity();
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rowMajorDiff = diff;
		Eigen::Map<Eigen::VectorXd>(pose_map.data() + (i - 1) * 9, 9) =
			Eigen::Map<Eigen::VectorXd>(rowMajorDiff.data(), 9);
	}

	Eigen::MatrixXd v_posed = v_shaped;
	if (poseBlendShapes_.cols() == pose_map.size())
	{
		Eigen::VectorXd pose_offset = poseBlendShapes_ * pose_map;
		v_posed += Eigen::Map<
			const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
			pose_offset.data(), N, 3);
	}
	else
	{
		std::cerr << "SMPLModel::getMesh - pose_blend_shapes dimension mismatch: "
				  << poseBlendShapes_.rows() << "x" << poseBlendShapes_.cols()
				  << " vs pose_map size " << pose_map.size() << std::endl;
	}

	// 4. Build LBS transforms (uses G_rot, G_trans from poseResult)

	// Convert to Matrix4d for LBS (with rest pose removal)
	std::vector<Eigen::Matrix4d> G(numJoints);
	for (int i = 0; i < numJoints; ++i)
	{
		G[i].setIdentity();
		G[i].block<3, 3>(0, 0) = poseResult.G_rot[i];
		G[i].block<3, 1>(0, 3) = poseResult.G_trans[i];
		// Apply rest pose removal
		Eigen::Matrix4d rest = Eigen::Matrix4d::Identity();
		rest.block<3, 1>(0, 3) = -J.row(i).transpose();
		G[i] = G[i] * rest;
	}

	// 5. Linear Blend Skinning
	Eigen::MatrixXd v_final(N, 3);

	for (int i = 0; i < N; ++i)
	{
		Eigen::Vector4d v_homo(v_posed(i, 0), v_posed(i, 1), v_posed(i, 2), 1.0f);
		Eigen::Vector4d v_sum = Eigen::Vector4d::Zero();

		for (int j = 0; j < numJoints; ++j)
		{
			double w = weights_(i, j);
			v_sum += w * (G[j] * v_homo);
		}

		v_final.row(i) = v_sum.head<3>().transpose();
	}

	// Output mesh
	mesh.vertices.reserve(N);
	for (int i = 0; i < N; ++i)
	{
		mesh.vertices.emplace_back(v_final(i, 0), v_final(i, 1), v_final(i, 2));
	}

	mesh.faces.reserve(faces_.rows());
	for (int i = 0; i < faces_.rows(); ++i)
	{
		mesh.faces.emplace_back(faces_(i, 0), faces_(i, 1), faces_(i, 2));
	}

	return mesh;
}
