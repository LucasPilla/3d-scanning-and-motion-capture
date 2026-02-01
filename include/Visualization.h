// Visualization
// Responsible for rendering and exporting results.

#pragma once

#include "CameraModel.h"
#include "PoseDetector.h"
#include "SMPLModel.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

class Visualization
{
public:
	Visualization(int width, int height, double fps, std::filesystem::path outputPath);

	void write(const cv::Mat &frame);
	
	void drawKeypoints(cv::Mat &frame, const std::vector<Point2D> &keypoints);

	// Draw SMPL mesh wireframe overlay on frame
	void drawMesh(cv::Mat &frame, const SMPLMesh &mesh, const CameraModel &camera,
				  const Eigen::Vector3d &globalT);

private:
	cv::VideoWriter writer;
};
