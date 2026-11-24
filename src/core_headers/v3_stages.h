// Authors: Team 8; implementation assisted by OpenAI Codex.
// V3 uses the same S6 warp as V1/V2 with a compressed crop stack.
#pragma once
#include "v1_stages.h"

// Blocking host-buffer interface, same contract as v1_warp_blend.
// Panorama and crop geometry match the original V1.
void v3_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);

namespace stage6_detail {
void launch_compressed_crop_rows(const int* heights, int width, int height,
                                 void* scratch, stage6_crop::Rect* rows);
}
