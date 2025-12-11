// FittingOptimizer
// -----------------
// Performs non-linear optimization (SMPLify-style) to fit SMPL parameters
// to 2D OpenPose detections using Ceres Solver.
// Responsibilities:
//  - Define cost functions for joint reprojection
//  - Initialize pose & shape
//  - Run optimization for each frame
// Used by:
//  - TemporalSmoother (after all per-frame fits are computed)
