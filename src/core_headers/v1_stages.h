// File: v1_stages.h
// Authors: Team 8
// Description: Stage interfaces for the basic CUDA pipeline (V1).
// External sources: none.
#pragma once
#include "common.h"

// S3-S4: Blocking host-buffer interfaces for the baseline global-memory CUDA stages.
// Invalid S3 input returns an empty output, matching V0. CUDA failures throw
// std::runtime_error; these functions never fall back to CPU computation.
void v1_describe(const Gray& blurred, const std::vector<Keypoint>& kps,
                 std::vector<Descriptor>& desc);
void v1_match(const std::vector<Descriptor>& a, const std::vector<Descriptor>& b,
              std::vector<Match>& matches);

// S6: CUDA inverse warp, feather blend, and valid crop.
// Invalid input returns an empty image. CUDA failures are logged to stderr.
void v1_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);
