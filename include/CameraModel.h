#pragma once

#include <Eigen/Dense>

struct CameraIntrinsics
{
    float fx;  // focal length in x (pixels)
    float fy;  // focal length in y (pixels)
    float cx;  // principal point x (pixels)
    float cy;  // principal point y (pixels)
};

class CameraModel
{
public:
    CameraModel() = default;

    CameraModel(float fx, float fy, float cx, float cy)
    {
        K_.fx = fx;
        K_.fy = fy;
        K_.cx = cx;
        K_.cy = cy;
    }

    explicit CameraModel(const CameraIntrinsics& intrinsics)
        : K_(intrinsics)
    {}

    const CameraIntrinsics& intrinsics() const { return K_; }
    CameraIntrinsics& intrinsics() { return K_; }

    // Pinhole projection: 3D point in camera coordinates -> 2D pixel
    // Assumes pointCam.z() > 0.
    Eigen::Vector2f project(const Eigen::Vector3f& pointCam) const
    {
        float xNorm = pointCam.x() / pointCam.z();
        float yNorm = pointCam.y() / pointCam.z();

        float u = K_.fx * xNorm + K_.cx;
        float v = K_.fy * yNorm + K_.cy;

        return Eigen::Vector2f(u, v);
    }

private:
    CameraIntrinsics K_{};
};