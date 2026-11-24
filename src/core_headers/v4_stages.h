// Authors: Team 8; implementation assisted by Antigravity.
// V4 inherits all of V3 (including V2 S4 and V3 S6 compressed crop) and optimizes S6 warp/blend.
#pragma once
#include "common.h"

void v4_warp_blend(const Image& A, const Image& B, const Homography& H, Image& pano);
