// TemporalSmoother.cpp
// Implements utilities for smoothing / regularizing SMPL parameters over time.
// Reduces jitter in per-frame reconstruction.
// Supports:
//   - simple filtering (moving average, Gaussian, Savitzky–Golay)
//   - temporal regularization terms used during optimization
//
// TODOs:
//   - implement smoothPoseSequence() and smoothShapeSequence()
//   - experiment with smoothing hyperparameters
//   - ensure body motion stays natural (no over-smoothing)

#include "TemporalSmoother.h"

TemporalSmoother::ParamSequence
TemporalSmoother::smoothPoseSequence(const ParamSequence& rawPoseSequence)
{
    if (rawPoseSequence.empty()) {
        return rawPoseSequence;
    }

    // A small centered window (e.g. 5 frames) works well for pose
    constexpr int kWindowSize = 5;
    return applyMovingAverage(rawPoseSequence, kWindowSize);
}

TemporalSmoother::ParamSequence
TemporalSmoother::smoothShapeSequence(const ParamSequence& rawShapeSequence)
{
    if (rawShapeSequence.empty()) {
        return rawShapeSequence;
    }

    // Shape varies more slowly, slightly larger window can be used if desired
    constexpr int kWindowSize = 7;
    return applyMovingAverage(rawShapeSequence, kWindowSize);
}

TemporalSmoother::ParamSequence
TemporalSmoother::applyMovingAverage(const ParamSequence& sequence,
                                     int windowSize)
{
    if (sequence.empty() || windowSize <= 1) {
        return sequence;
    }

    const int T = static_cast<int>(sequence.size());
    const int D = static_cast<int>(sequence.front().size());
    TemporalSmoother::ParamSequence smoothed(
        T, std::vector<double>(D, 0.0)
    );

    const int half = windowSize / 2;

    for (int t = 0; t < T; ++t) {
        const int start = std::max(0, t - half);
        const int end   = std::min(T - 1, t + half);
        const int count = end - start + 1;

        for (int d = 0; d < D; ++d) {
            double sum = 0.0;
            for (int k = start; k <= end; ++k) {
                // For inconsistent dimensions
                if (d < static_cast<int>(sequence[k].size())) {
                    sum += sequence[k][d];
                }
            }
            smoothed[t][d] = sum / static_cast<double>(count);
        }
    }

    return smoothed;
}

// Keep Gaussian / Savitzky–Golay as TODO stubs for now
TemporalSmoother::ParamSequence
TemporalSmoother::applyGaussianFilter(const ParamSequence& sequence,
                                      double /*sigma*/,
                                      int /*kernelRadius*/)
{
    return sequence;
}

TemporalSmoother::ParamSequence
TemporalSmoother::applySavitzkyGolay(const ParamSequence& sequence,
                                     int /*windowSize*/,
                                     int /*polyOrder*/)
{
    return sequence;
}