// PoseDetector
// -------------
// Wraps OpenPose for extracting 2D human pose keypoints from video frames.
// Responsibilities:
//  - Initialize OpenPose with required models
//  - Process each frame with OpenPose
//  - Output 2D keypoints as a simple array/vector
// Used by:
//  - FittingOptimizer (which fits SMPL to the 2D joints)
