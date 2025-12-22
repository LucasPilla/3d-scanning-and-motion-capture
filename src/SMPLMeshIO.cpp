#include "SMPLMeshIO.h"
#include <fstream>
#include <iomanip>
#include <iostream>

bool SMPLMeshIO::saveOBJ(const std::string& filename, const SMPLMesh& mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for mesh export: " << filename << std::endl;
        return false;
    }

    // Set precision for high-quality vertex coordinates
    file << std::fixed << std::setprecision(6);

    // Write Vertices (v x y z)
    for (const auto& v : mesh.vertices) {
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }

    // Write Faces (f v1 v2 v3)
    // Note: OBJ indices are 1-based, while Eigen/C++ are 0-based.
    for (const auto& f : mesh.faces) {
        file << "f " << f.x() + 1 << " " << f.y() + 1 << " " << f.z() + 1 << "\n";
    }

    file.close();
    return true;
}

bool SMPLMeshIO::saveJointsOBJ(const std::string& filename, const std::vector<Eigen::Vector3f>& joints) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    // Export joints as single vertices. 
    // In MeshLab, these will appear as a point cloud.
    for (const auto& j : joints) {
        file << "v " << j.x() << " " << j.y() << " " << j.z() << "\n";
    }

    file.close();
    return true;
}