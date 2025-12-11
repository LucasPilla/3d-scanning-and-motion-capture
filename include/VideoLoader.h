// VideoLoader
// ------------
// Loads an RGB video from disk and provides sequential frame access.
// Responsibilities:
//  - Open input video file
//  - Extract frames one-by-one
//  - Optionally downsample (skip frames for faster processing)
//  - Convert frames to OpenCV Mat format
// Used by:
//  - PoseDetector (which runs OpenPose on each frame)
