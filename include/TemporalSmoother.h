// TemporalSmoother
// -----------------
// Applies temporal filtering on SMPL pose and/or shape parameters to reduce jitter.
// Responsibilities:
//  - Smooth predicted SMPL parameters across time
//  - Implement simple filters (moving average, Gaussian) OR
//    temporal optimization (frame-to-frame penalties)
// Used by:
//  - Visualization (for producing smooth 3D animations)
