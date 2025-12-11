// PoseDetector.cpp
// Implementation of the PoseDetector class (OpenPose wrapper).
// Contains the code for:
//   - initializing the OpenPose wrapper
//   - running pose estimation on each frame
//   - converting OpenPose output to our internal keypoint format
//
// Main TODOs:
//   - configure OpenPose model paths
//   - implement detectKeypoints(frame)
//   - handle multiple people (for now we will use only person 0)
