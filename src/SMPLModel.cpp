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

SMPLMesh SMPLModel::getMesh() const
{
    SMPLMesh mesh;

    if (!loaded_) {
        // Return empty mesh if the model has not been loaded yet.
        return mesh;
    }

    const int numVertices = static_cast<int>(templateVertices_.rows());
    const int numFaces    = static_cast<int>(faces_.rows());

    mesh.vertices.reserve(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        mesh.vertices.emplace_back(
            templateVertices_(i, 0),
            templateVertices_(i, 1),
            templateVertices_(i, 2)
        );
    }

    mesh.faces.reserve(numFaces);
    for (int i = 0; i < numFaces; ++i) {
        mesh.faces.emplace_back(
            faces_(i, 0),
            faces_(i, 1),
            faces_(i, 2)
        );
    }

    return mesh;
}