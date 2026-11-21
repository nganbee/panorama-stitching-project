// Authors: Team 8; implementation assisted by OpenAI Codex.
// V3 experiment: source-pixel texture fetches, preserving V1's S6 arithmetic.
#pragma once
#include "common.h"

// Blocking host-buffer interface, same contract as v1_warp_blend.
// Manual double-precision bilinear, feathering and crop match the original V1.
void v3_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);
