#pragma once
#include <string>
#include <vector>
#include "SMPLModel.h"

class SMPLMeshIO {
public:
    /**
     * @brief Exports a SMPLMesh to a Wavefront OBJ file.
     * @param filename Path to save (e.g., "debug_frame_0.obj")
     * @param mesh The mesh data containing vertices and faces
     */
    static bool saveOBJ(const std::string& filename, const SMPLMesh& mesh);

    /**
     * @brief Exports joint locations as a point cloud or small markers in OBJ format.
     */
    static bool saveJointsOBJ(const std::string& filename, const std::vector<Eigen::Vector3f>& joints);
};