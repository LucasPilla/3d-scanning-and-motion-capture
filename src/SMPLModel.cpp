// SMPLModel.cpp
// Implements the SMPLModel interface defined in SMPLModel.h.
// This file is responsible for:
//   - loading SMPL model files (blend shapes, joint regressor, etc.)
//   - implementing forward() to compute a 3D mesh from pose + shape
//   - returning joint positions for optimization
//
// TODOs:
//   - load model data (typically .npz or .pkl converted to C++)
//   - implement pose blend shapes, shape blend shapes
//   - compute final skinned vertices

#include "SMPLModel.h"

#include <fstream>
#include <iostream>

// Single-header JSON library (see note in SMPLModel.h).
#include <nlohmann/json.hpp>

using nlohmann::json;

bool SMPLModel::loadFromJson(const std::string& jsonPath)
{
    loaded_ = false;

    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        std::cerr << "SMPLModel::loadFromJson - cannot open file: " << jsonPath << "\n";
        return false;
    }

    json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        std::cerr << "SMPLModel::loadFromJson - JSON parse error: " << e.what() << "\n";
        return false;
    }

    try {
        // -------- vertices_template (N, 3) --------
        const auto& vt = j.at("vertices_template");
        const int numVertices = static_cast<int>(vt.size());
        if (numVertices == 0 || vt.at(0).size() != 3) {
            std::cerr << "SMPLModel::loadFromJson - invalid vertices_template shape\n";
            return false;
        }

        templateVertices_.resize(numVertices, 3);
        for (int i = 0; i < numVertices; ++i) {
            for (int c = 0; c < 3; ++c) {
                templateVertices_(i, c) = static_cast<float>(vt[i][c]);
            }
        }

        // -------- face_indices (F, 3) --------
        const auto& faces = j.at("face_indices");
        const int numFaces = static_cast<int>(faces.size());
        faces_.resize(numFaces, 3);
        for (int i = 0; i < numFaces; ++i) {
            for (int c = 0; c < 3; ++c) {
                // preprocess.py stores 1-based indices; convert to 0-based for C++
                int idx = static_cast<int>(faces[i][c]) - 1;
                faces_(i, c) = idx;
            }
        }

        // -------- shape_blend_shapes (N, 3, numShapeCoeffs) --------
        const auto& shapeBs = j.at("shape_blend_shapes");
        const int numShapeVerts = static_cast<int>(shapeBs.size());       // should be N
        const int numShapeComps = static_cast<int>(shapeBs[0].size());    // should be 3
        const int numShapeCoeffs = static_cast<int>(shapeBs[0][0].size()); // e.g. 10

        if (numShapeVerts != numVertices || numShapeComps != 3) {
            std::cerr << "SMPLModel::loadFromJson - invalid shape_blend_shapes shape\n";
            return false;
        }

        shapeBlendShapes_.resize(numVertices * 3, numShapeCoeffs);
        for (int v = 0; v < numVertices; ++v) {
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < numShapeCoeffs; ++k) {
                    int row = 3 * v + c;
                    shapeBlendShapes_(row, k) = static_cast<float>(shapeBs[v][c][k]);
                }
            }
        }

        // -------- pose_blend_shapes (N, 3, numPoseCoeffs) --------
        const auto& poseBs = j.at("pose_blend_shapes");
        const int numPoseVerts = static_cast<int>(poseBs.size());
        const int numPoseComps = static_cast<int>(poseBs[0].size());
        const int numPoseCoeffs = static_cast<int>(poseBs[0][0].size()); // e.g. 207

        if (numPoseVerts != numVertices || numPoseComps != 3) {
            std::cerr << "SMPLModel::loadFromJson - invalid pose_blend_shapes shape\n";
            return false;
        }

        poseBlendShapes_.resize(numVertices * 3, numPoseCoeffs);
        for (int v = 0; v < numVertices; ++v) {
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < numPoseCoeffs; ++k) {
                    int row = 3 * v + c;
                    poseBlendShapes_(row, k) = static_cast<float>(poseBs[v][c][k]);
                }
            }
        }

        // -------- joint_regressor (numJoints, N) --------
        const auto& jr = j.at("joint_regressor");
        const int numJoints = static_cast<int>(jr.size());
        if (numJoints == 0 || static_cast<int>(jr[0].size()) != numVertices) {
            std::cerr << "SMPLModel::loadFromJson - invalid joint_regressor shape\n";
            return false;
        }

        jointRegressor_.resize(numJoints, numVertices);
        for (int jIdx = 0; jIdx < numJoints; ++jIdx) {
            for (int v = 0; v < numVertices; ++v) {
                jointRegressor_(jIdx, v) = static_cast<float>(jr[jIdx][v]);
            }
        }
        // -------- weights (N, numJoints) --------
        const auto& w = j.at("weights");
        const int numWeightVerts = static_cast<int>(w.size());
        const int numWeightJoints = static_cast<int>(w[0].size());

        if (numWeightVerts != numVertices || numWeightJoints != numJoints) {
            std::cerr << "SMPLModel::loadFromJson - invalid weights shape\n";
            return false;
        }

        weights_.resize(numWeightVerts, numWeightJoints);
        for (int v = 0; v < numWeightVerts; ++v) {
            for (int k = 0; k < numWeightJoints; ++k) {
                weights_(v, k) = static_cast<float>(w[v][k]);
            }
        }

        // -------- kinematic_tree (2, numJoints) --------
        const auto& kt = j.at("kinematic_tree");
        const int ktRows = static_cast<int>(kt.size());
        const int ktCols = static_cast<int>(kt[0].size());

        if (ktRows != 2 || ktCols != numJoints) {
            std::cerr << "SMPLModel::loadFromJson - invalid kinematic_tree shape\n";
            return false;
        }

        kinematicTree_.resize(ktRows, ktCols);
        for (int r = 0; r < ktRows; ++r) {
            for (int c = 0; c < ktCols; ++c) {
                kinematicTree_(r, c) = static_cast<int>(kt[r][c]);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "SMPLModel::loadFromJson - error while reading arrays: " << e.what() << "\n";
        return false;
    }

    loaded_ = true;
    std::cout << "SMPLModel::loadFromJson - loaded model from " << jsonPath << "\n";
    return true;
}

void SMPLModel::setPose(const std::vector<double>& poseParams)
{
    poseParams_.resize(static_cast<int>(poseParams.size()));
    for (int i = 0; i < static_cast<int>(poseParams.size()); ++i) {
        poseParams_[i] = static_cast<float>(poseParams[i]);
    }
}

void SMPLModel::setShape(const std::vector<double>& shapeParams)
{
    shapeParams_.resize(static_cast<int>(shapeParams.size()));
    for (int i = 0; i < static_cast<int>(shapeParams.size()); ++i) {
        shapeParams_[i] = static_cast<float>(shapeParams[i]);
    }
}

//For pose blend shapes
Eigen::Matrix3f SMPLModel::rodrigues(const Eigen::Vector3f& r) const
{
    float theta = r.norm();
    if (theta < 1e-8f) {
        return Eigen::Matrix3f::Identity();
    }

    Eigen::Vector3f k = r / theta;

    Eigen::Matrix3f K;
    K <<     0, -k.z(),  k.y(),
          k.z(),     0, -k.x(),
         -k.y(),  k.x(),     0;

    return Eigen::Matrix3f::Identity()
         + std::sin(theta) * K
         + (1 - std::cos(theta)) * (K * K);
}

//For joint regression
Eigen::MatrixXf SMPLModel::computeJoints(const Eigen::MatrixXf& vertices) const
{
    // jointRegressor_: (24, N)
    // vertices: (N, 3)
    return jointRegressor_ * vertices;
}

//For forward kinematics
std::vector<Eigen::Matrix4f> SMPLModel::computeGlobalTransforms( const Eigen::MatrixXf& J, const std::vector<Eigen::Matrix3f>& rotations) const
{
    const int numJoints = static_cast<int>(rotations.size());
    std::vector<Eigen::Matrix4f> G(numJoints);

    // kinematic tree: parents are in row 0
    Eigen::VectorXi parents = kinematicTree_.row(0);

    // Root
    G[0].setIdentity();
    G[0].block<3,3>(0,0) = rotations[0];
    G[0].block<3,1>(0,3) = J.row(0).transpose();

    for (int i = 1; i < numJoints; ++i) {
        int p = parents(i);

        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3,3>(0,0) = rotations[i];
        T.block<3,1>(0,3) = (J.row(i) - J.row(p)).transpose();

        G[i] = G[p] * T;
    }

    // Remove rest pose
    for (int i = 0; i < numJoints; ++i) {
        Eigen::Matrix4f rest = Eigen::Matrix4f::Identity();
        rest.block<3,1>(0,3) = -J.row(i).transpose();
        G[i] = G[i] * rest;
    }

    return G;
}



SMPLMesh SMPLModel::getMesh() const
{
    SMPLMesh mesh;
    if (!loaded_) return mesh;

    const int N         = templateVertices_.rows();
    const int numJoints = jointRegressor_.rows();

    // 1. Skip shape and pose offsets for now: just use template vertices
    Eigen::MatrixXf v_shaped = templateVertices_;
    Eigen::MatrixXf v_posed  = v_shaped;

    // 2. Joint regression on template
    Eigen::MatrixXf J = computeJoints(v_shaped);

    // 3. Identity rotations (no pose)
    std::vector<Eigen::Matrix3f> rotations(numJoints, Eigen::Matrix3f::Identity());

    // 4. Forward kinematics
    auto G = computeGlobalTransforms(J, rotations);

    // 5. Linear Blend Skinning
    Eigen::MatrixXf v_final(N, 3);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector4f v_homo(v_posed(i, 0), v_posed(i, 1), v_posed(i, 2), 1.0f);
        Eigen::Vector4f v_sum = Eigen::Vector4f::Zero();

        for (int j = 0; j < numJoints; ++j) {
            float w = (i < weights_.rows() && j < weights_.cols())
                          ? weights_(i, j)
                          : 0.0f;
            v_sum += w * (G[j] * v_homo);
        }

        v_final.row(i) = v_sum.head<3>().transpose();
    }

    mesh.vertices.reserve(N);
    for (int i = 0; i < N; ++i)
        mesh.vertices.emplace_back(v_final(i, 0), v_final(i, 1), v_final(i, 2));

    mesh.faces.reserve(faces_.rows());
    for (int i = 0; i < faces_.rows(); ++i)
        mesh.faces.emplace_back(faces_(i, 0), faces_(i, 1), faces_(i, 2));

    return mesh;
}