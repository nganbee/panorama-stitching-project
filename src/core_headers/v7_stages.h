// File: v7_stages.h
// Authors: Team 8 (Thụ: V7 owner)
// Description: V7 stage interface: S5 RANSAC with one warp per hypothesis.
// External sources: none.
//
// V7 changes only how S5 computes inlier scores. Everything else (S1-S4, S6,
// GPU data residency) is inherited unchanged from the latest integrated
// version (V5 device-pointer stages in v5_stages.h). The function below has the
// same contract as v5::ransac_homography: device keypoints/matches in, device
// homography result out, plus a host copy for status/reporting.
//
// Kept identical to V0/V1/V5: deterministic MT19937 sample sequence (seed
// params::kRansacSeed), params::kRansacIters = 2000 hypotheses, threshold
// params::kRansacThresh, normalized 4-point DLT, best-hypothesis tie-break
// (more inliers, then lower summed error, then earliest iteration) and the
// final least-squares refinement on all inliers.
#pragma once
#include "common.h"
#include "v5_stages.h"

namespace v7 {
// S5: one CUDA thread estimates each 4-point hypothesis (same as V1), then one
// warp scores it: 32 lanes test 32 matches at a time, __ballot_sync/__popc
// count the inliers, and the summed error is accumulated in match order so the
// result is bit-identical to V5. The best-hypothesis search and the inlier
// selection/count of the refinement use the same warp scheme; the refined
// least-squares DLT itself is unchanged.
bool ransac_homography(const Keypoint* kpA, int na, const Keypoint* kpB, int nb,
                       const Match* matches, int n,
                       V5HomographyResult* device_result, V5HomographyResult& host_result);
} // namespace v7
