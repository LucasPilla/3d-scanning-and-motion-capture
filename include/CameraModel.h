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
        K_.fx = 1000.0;
        K_.fy = 1000.0;
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