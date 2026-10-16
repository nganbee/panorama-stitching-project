// File: v0_stages.h
// Authors: Team 8; maintenance assisted by OpenAI Codex.
// Description: Stage interfaces for the sequential pipeline.
// External sources: none.
#pragma once
#include "common.h"

// S1/S2 callers must supply correctly sized images/Gray buffers; run_v0
// validates its RGB inputs. S3/S5/S6 reject malformed inputs explicitly.

// Stage 1: Gray Scale, Gaussian Blur
void v0_rgb_to_gray(const Image& in, Gray& gray);
void v0_gaussian_blur(const Gray& in, Gray& out);

// Stage 2: Harris Corner Detection, Non-Maximum Suppression, Top-K
void v0_harris_response(const Gray& blurred, Gray& response);
void v0_nms_topk(const Gray& response, std::vector<Keypoint>& kps);

// S3
void v0_describe(const Gray& blurred, const std::vector<Keypoint>& kps,
                 std::vector<Descriptor>& desc);

// S4
void v0_match(const std::vector<Descriptor>& dA, const std::vector<Descriptor>& dB,
              std::vector<Match>& matches);

// S5
bool v0_ransac_homography(const std::vector<Keypoint>& kpA, const std::vector<Keypoint>& kpB,
                          const std::vector<Match>& matches, Homography& H, int& nInliers);

// S6: Returns an empty panorama for invalid images or unsafe homographies.
void v0_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);

bool v0_selftest_stage12();  // returns true if all tests pass
