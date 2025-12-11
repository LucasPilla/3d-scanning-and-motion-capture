// TemporalSmoother.cpp
// Implements smoothing of SMPL parameters over time.
// Reduces jitter in per-frame reconstruction.
// Supports:
//   - simple filtering (moving average, Gaussian)
//   - optional temporal regularization optimization
//
// TODOs:
//   - implement smoothPoseSequence() and smoothShapeSequence()
//   - experiment with smoothing hyperparameters
//   - ensure body motion stays natural (no over-smoothing)
