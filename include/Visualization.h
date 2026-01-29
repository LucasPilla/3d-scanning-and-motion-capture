// Visualization
// Responsible for rendering and exporting results.

#pragma once
#include "CameraModel.h"
#include "PoseDetector.h"
#include "SMPLModel.h"
#include <opencv2/opencv.hpp>

class Visualization
{
public:
	Visualization(int width, int height, double fps);

	void write(const cv::Mat &frame);
	void drawKeypoints(cv::Mat &frame, const std::vector<Point2D> &keypoints);

	// Generic 2D joint drawing (for SMPL projections, etc.)
	void drawJoints(cv::Mat &frame, const std::vector<Point2D> &joints,
					const cv::Scalar &color, int radius = 4);

	// Draw SMPL mesh wireframe overlay on frame
	void drawMesh(cv::Mat &frame, const SMPLMesh &mesh, const CameraModel &camera,
				  const Eigen::Vector3d &globalT,
				  const cv::Scalar &color = cv::Scalar(0, 255, 255),
				  int lineThickness = 1);

private:
	cv::VideoWriter writer;
};
