#pragma once

#include <string>
#include <vector>
#include "SMPLModel.h"

class SMPLMeshIO {
public:
    /**
     * @brief Exports the SMPLMesh (vertices and faces) to Wavefront OBJ format.
     * @return true if successful.
     */
    static bool saveOBJ(const std::string& filename, const SMPLMesh& mesh);

    /**
     * @brief Exports a list of 3D points (joints) as a separate OBJ.
     * Helpful for checking if joints are correctly regressed inside the mesh.
     */
    static bool saveJointsOBJ(const std::string& filename, const std::vector<Eigen::Vector3f>& joints);
};