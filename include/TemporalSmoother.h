// TemporalSmoother
// -----------------
// Provides utilities for temporal filtering / regularization of SMPL pose
// and shape parameters.
// Responsibilities:
//  - Smooth predicted SMPL parameters across time
//  - Provide simple filters (moving average, Gaussian, Savitzky–Golay) that can
//    be used to build temporal regularization terms inside the optimizer.
// Used by:
//  - FittingOptimizer (for temporal regularization during fitting)
//  - (Optionally) Visualization / analysis for offline smoothing experiments.

#pragma once

#include <vector>

// Temporal smoothing of SMPL parameters (pose & shape)
//
// We represent a sequence of parameters as:
//   - outer dimension: time (frames)
//   - inner dimension: parameter vector for a single frame
//     e.g. pose: [72], shape: [10]
class TemporalSmoother
{
public:
    using ParamSequence = std::vector<std::vector<double>>;

    TemporalSmoother() = default;

    // High-level entry point to smooth a sequence of pose parameters
    //
    // NOTE: For now this will just forward to a placeholder implementation
    // TODO: choose method (moving average / Gaussian / Savitzky–Golay)
    // based on configuration or experiment flags
    ParamSequence smoothPoseSequence(const ParamSequence& rawPoseSequence);

    // High-level entry point to smooth a sequence of shape parameters
    //
    // Shape is typically much lower-dimensional and slower-varying than pose
    // TODO (future): consider weaker smoothing or even keeping shape constant
    ParamSequence smoothShapeSequence(const ParamSequence& rawShapeSequence);

private:
    // ------- Placeholder filter implementations (no real math yet) -------

    // TODO:
    // Implement a simple causal/centered moving average filter over time
    //
    // Example behavior:
    //   - windowSize = 3 -> average of [t-1, t, t+1]
    //   - handle boundaries by shrinking the window or mirroring
    ParamSequence applyMovingAverage(const ParamSequence& sequence,
                                     int windowSize);

    // TODO:
    // Implement a 1D Gaussian filter along the time axis
    //
    // Example behavior:
    //   - build a normalized Gaussian kernel with std-dev = sigma
    //   - convolve over time for each parameter dimension independently
    ParamSequence applyGaussianFilter(const ParamSequence& sequence,
                                      double sigma,
                                      int kernelRadius);

    // TODO:
    // Implement a Savitzky–Golay filter along the time axis
    //
    // Example behavior:
    //   - fit low-order polynomials (polyOrder) in a sliding window
    //   - evaluate smoothed value at the window center
    ParamSequence applySavitzkyGolay(const ParamSequence& sequence,
                                     int windowSize,
                                     int polyOrder);
};