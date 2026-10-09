// stage-level interface of V0, used to implement the full pipeline in v0_stages.cpp
#pragma once
#include "common.h"

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

// S6
void v0_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);

bool v0_selftest_stage12();  // returns true if all tests pass