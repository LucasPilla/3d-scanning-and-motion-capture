// Visualization.cpp
// Implements visualization utilities for drawing detected 
// keypoints, 3D joints and 3D mesh over the input frame

#include "Visualization.h"
#include "PoseDetector.h"

Visualization::Visualization(int width, int height, double fps, std::filesystem::path outputPath)
{
	writer.open(outputPath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
				cv::Size(width, height), true);

	if (!writer.isOpened())
	{
		std::cerr << "ERROR: Could not open video writer!" << std::endl;
	}
	else
	{
		std::cout << "Video writer opened. " << std::endl;
	}
}

void Visualization::write(const cv::Mat &frame)
{
	if (!frame.empty() && writer.isOpened())
	{
		writer.write(frame);
	}
}

void Visualization::drawKeypoints(cv::Mat &frame,
								  const std::vector<Point2D> &keypoints)
{
	// This function expects 25 keypoints, according to BODY_25 model
	if (keypoints.size() != 25)
	{
		return;
	}

	// BODY_25 keypoint connections (skeleton)
	// Each pair represents a connection between two keypoints
	const std::vector<std::pair<int, int>> connections = {
		{1, 8}, // Neck → MidHip
		{1, 2},
		{2, 3},
		{3, 4}, // Right arm
		{1, 5},
		{5, 6},
		{6, 7}, // Left arm
		{8, 9},
		{9, 10},
		{10, 11}, // Right leg
		{8, 12},
		{12, 13},
		{13, 14}, // Left leg
		{1, 0},	  // Neck → Nose
		{0, 15},
		{15, 17}, // Right face
		{0, 16},
		{16, 18}, // Left face
		{14, 19},
		{19, 20},
		{14, 21}, // Left foot
		{11, 22},
		{22, 23},
		{11, 24} // Right foot
	};

	// Colors for visualization
	const cv::Scalar skeleton_color = cv::Scalar(0, 255, 0); // Green for skeleton
	const cv::Scalar keypoint_color = cv::Scalar(0, 0, 255); // Red for keypoints

	const int keypoint_radius = 4;
	const int skeleton_thickness = 2;

	// Draw skeleton connections
	for (const auto &connection : connections)
	{

		int idx1 = connection.first;
		int idx2 = connection.second;

		const Point2D pt1 = keypoints[idx1];
		const Point2D pt2 = keypoints[idx2];

		if (pt1.x > 0 && pt1.y > 0 && pt1.score > 0.0 && pt2.x > 0 && pt2.y > 0 &&
			pt2.score > 0.0)
		{
			cv::line(frame, cv::Point(pt1.x, pt1.y), cv::Point(pt2.x, pt2.y),
					 skeleton_color, skeleton_thickness);
		}
	}

	// Draw keypoints
	for (size_t i = 0; i < keypoints.size(); ++i)
	{

		const Point2D pt = keypoints[i];

		// Skip invalid keypoints
		if (pt.x <= 0 || pt.y <= 0 || pt.score <= 0.0)
		{
			continue;
		}

		cv::circle(frame, cv::Point(pt.x, pt.y), keypoint_radius, keypoint_color,
				   -1);
	}
}

void Visualization::drawWireframe(cv::Mat &frame, const SMPLMesh &mesh,
							 const CameraModel &camera,
							 const Eigen::Vector3d &globalT,
							 cv::Scalar color)
{
	int lineThickness = 1;

	const float fx = camera.intrinsics().fx;
	const float fy = camera.intrinsics().fy;
	const float cx = camera.intrinsics().cx;
	const float cy = camera.intrinsics().cy;

	const int frameW = frame.cols;
	const int frameH = frame.rows;

	// Project all vertices to 2D
	std::vector<cv::Point> projectedVerts(mesh.vertices.size());
	std::vector<bool> validVerts(mesh.vertices.size(), false);

	for (size_t i = 0; i < mesh.vertices.size(); ++i)
	{
		const Eigen::Vector3f &v = mesh.vertices[i];

		// Apply global transform (camera-to-body alignment)
		Eigen::Vector3d pWorld(v.x(), v.y(), v.z());
		Eigen::Vector3d pCam = pWorld + globalT;

		// Skip if behind camera
		if (pCam.z() < 0.1)
			continue;

		// Project using pinhole camera
		float u = fx * (pCam.x() / pCam.z()) + cx;
		float v_coord = fy * (pCam.y() / pCam.z()) + cy;

		// Check bounds
		if (u >= 0 && u < frameW && v_coord >= 0 && v_coord < frameH)
		{
			projectedVerts[i] =
				cv::Point(static_cast<int>(u), static_cast<int>(v_coord));
			validVerts[i] = true;
		}
	}

	// Draw triangle edges (wireframe)
	for (const auto &face : mesh.faces)
	{
		int i0 = face.x();
		int i1 = face.y();
		int i2 = face.z();

		// Draw edges only if both vertices are valid
		if (validVerts[i0] && validVerts[i1])
		{
			cv::line(frame, projectedVerts[i0], projectedVerts[i1], color,
					 lineThickness, cv::LINE_AA);
		}
		if (validVerts[i1] && validVerts[i2])
		{
			cv::line(frame, projectedVerts[i1], projectedVerts[i2], color,
					 lineThickness, cv::LINE_AA);
		}
		if (validVerts[i2] && validVerts[i0])
		{
			cv::line(frame, projectedVerts[i2], projectedVerts[i0], color,
					 lineThickness, cv::LINE_AA);
		}
	}
}

void Visualization::drawMesh(cv::Mat &frame, const SMPLMesh &mesh,
                             const CameraModel &camera,
                             const Eigen::Vector3d &globalT,
                             cv::Scalar color)
{
    const auto& K = camera.intrinsics();
    const Eigen::Vector3f T = globalT.cast<float>();
    const int numVerts = mesh.vertices.size();

    std::vector<cv::Point> points2d(numVerts);
    std::vector<float> depths(numVerts);
    std::vector<uint8_t> isVisible(numVerts, 0); 

    // Project all vertices
    for (int i = 0; i < numVerts; ++i) {
        Eigen::Vector3f v = mesh.vertices[i] + T;
        
        if (v.z() <= 0.1f) continue; // Skip if behind camera

        float invZ = 1.0f / v.z();
        points2d[i].x = static_cast<int>(K.fx * v.x() * invZ + K.cx);
        points2d[i].y = static_cast<int>(K.fy * v.y() * invZ + K.cy);
        depths[i] = v.z();
        isVisible[i] = 1;
    }

    // Build draw list 
    std::vector<std::pair<float, int>> drawList;
    drawList.reserve(mesh.faces.size());

    for (int i = 0; i < (int)mesh.faces.size(); ++i) {
        const auto& f = mesh.faces[i];
        
        // Skip if any vertex is invalid
        if (!isVisible[f.x()] || !isVisible[f.y()] || !isVisible[f.z()]) continue;

        const cv::Point& p0 = points2d[f.x()];
        const cv::Point& p1 = points2d[f.y()];
        const cv::Point& p2 = points2d[f.z()];

        // 2D Cross Product for Backface Culling: (p1-p0) x (p2-p0)
        float cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        
        if (cross < 0) { 
            float zSum = depths[f.x()] + depths[f.y()] + depths[f.z()];
            drawList.push_back({zSum, i});
        }
    }

    // Sort faces by depth
    std::sort(drawList.begin(), drawList.end(), std::greater<std::pair<float, int>>());

    // Render
    const Eigen::Vector3f lightDir(0.0f, 0.0f, -1.0f);

    for (const auto& item : drawList) {
        const auto& f = mesh.faces[item.second];
        
        // Calculate Normal for shading
        Eigen::Vector3f v0 = mesh.vertices[f.x()];
        Eigen::Vector3f v1 = mesh.vertices[f.y()];
        Eigen::Vector3f v2 = mesh.vertices[f.z()];
        Eigen::Vector3f normal = (v1 - v0).cross(v2 - v0).normalized();

        float intensity = std::max(0.1f, normal.dot(lightDir));
        
        cv::Point tri[3] = { points2d[f.x()], points2d[f.y()], points2d[f.z()] };
        cv::fillConvexPoly(frame, tri, 3, color * intensity, cv::LINE_AA);
    }
}