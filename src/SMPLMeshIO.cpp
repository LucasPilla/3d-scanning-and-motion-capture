#include "SMPLMeshIO.h"
#include <fstream>
#include <iomanip>
#include <iostream>

bool SMPLMeshIO::saveOBJ(const std::string& filename, const SMPLMesh& mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for export: " << filename << std::endl;
        return false;
    }

    // Set precision for floating point coordinates
    file << std::fixed << std::setprecision(6);

    // Write Vertices
    for (const auto& v : mesh.vertices) {
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }

    // Write Faces (OBJ indices start at 1)
    for (const auto& f : mesh.faces) {
        file << "f " << f.x() + 1 << " " << f.y() + 1 << " " << f.z() + 1 << "\n";
    }

    file.close();
    return true;
}

bool SMPLMeshIO::saveJointsOBJ(const std::string& filename, const std::vector<Eigen::Vector3f>& joints) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    // Exporting joints as simple vertices (points)
    // In MeshLab, you may need to enable "Point Size" to see them clearly
    for (const auto& j : joints) {
        file << "v " << j.x() << " " << j.y() << " " << j.z() << "\n";
    }
    
    file.close();
    return true;
}