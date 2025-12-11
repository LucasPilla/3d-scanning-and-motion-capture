// SMPLModel
// ----------
// Interface for the SMPL (Skinned Multi-Person Linear) parametric body model.
// Responsibilities:
//  - Load SMPL model data (shape blend shapes, pose blend shapes, joint regressor)
//  - Convert pose + shape parameters into a 3D mesh
//  - Provide 3D joint locations for optimization
// Used by:
//  - FittingOptimizer
//  - Visualization
