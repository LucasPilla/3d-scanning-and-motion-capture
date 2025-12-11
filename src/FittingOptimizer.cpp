// FittingOptimizer.cpp
// Implements the SMPL fitting optimization pipeline.
// Uses Ceres Solver to minimize reprojection error between:
//   - projected 3D SMPL joints
//   - 2D OpenPose keypoints
//
// Contains:
//   - construction of Ceres problem
//   - cost functions for joint reprojection
//   - optimization loop per frame
//
// TODOs:
//   - define cost residuals
//   - implement solveFrame() for per-frame fitting
//   - add initialization heuristics for pose/shape
