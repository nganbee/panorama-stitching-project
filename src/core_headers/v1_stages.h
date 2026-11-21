// File: v1_stages.h
// Authors: Team 8
// Description: Stage interfaces for the basic CUDA pipeline (V1).
// External sources: none.
#pragma once
#include "common.h"

// S1-S2: blocking host-buffer interfaces for the basic CUDA implementation.
// V1 intentionally keeps the wrappers simple: each call owns its device
// buffers and synchronizes before returning. Later versions can introduce a
// persistent device workspace without changing the numerical contract.
void v1_rgb_to_gray(const Image& in, Gray& gray);
void v1_gaussian_blur(const Gray& in, Gray& out);
void v1_harris_response(const Gray& blurred, Gray& response);
void v1_nms_topk(const Gray& response, std::vector<Keypoint>& keypoints);

// S3-S4: Blocking host-buffer interfaces for the baseline global-memory CUDA stages.
// Invalid S3 input returns an empty output, matching V0. CUDA failures throw
// std::runtime_error; these functions never fall back to CPU computation.
void v1_describe(const Gray& blurred, const std::vector<Keypoint>& kps,
                 std::vector<Descriptor>& desc);
void v1_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b,
              std::vector<Match>& matches);

// S5: GPU match validation, sampling, normalized DLT, scoring and refinement.
// H maps B -> A. Invalid/degenerate data returns false; CUDA failures throw.
bool v1_ransac_homography(const std::vector<Keypoint>& kpA, const std::vector<Keypoint>& kpB,
                          const std::vector<Match>& matches, Homography& H, int& nInliers);

// S6: GPU transform/canvas preparation, inverse warp, feather blend, and crop.
// Invalid input returns an empty image. CUDA failures throw std::runtime_error.
void v1_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);
