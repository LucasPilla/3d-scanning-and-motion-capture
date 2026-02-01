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

    CameraModel(double frameWidth, double frameHeight)
    {
        // Those values were defined to approximate MPI INF 3DHP
        // camera intrinsics. The official SMPLify implementation
        // uses a hard-coded value of 5000.
        K_.fx = 1500.0;
        K_.fy = 1500.0;
        K_.cx = frameWidth / 2.0;
        K_.cy = frameHeight / 2.0;
        frameWidth_ = frameWidth;
        frameHeight_ = frameHeight;
    }

    const CameraIntrinsics& intrinsics() const { return K_; }
    double getFrameWidth() const { return frameWidth_; }
    double getFrameHeight() const { return frameHeight_; }

private:
    CameraIntrinsics K_{};
    double frameWidth_;
    double frameHeight_;
};